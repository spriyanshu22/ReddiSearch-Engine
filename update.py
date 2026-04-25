import os
import math
import requests
import psycopg2
from dotenv import load_dotenv
import time

# --- NLP Preprocessing Imports ---
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

load_dotenv()

# Initialize NLP tools once at module level (not per-post) for performance
STEMMER    = SnowballStemmer("english")
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):
    """
    Converts raw text into a list of cleaned, stemmed tokens.
    This is the standard NLP pipeline:
      1. Tokenize  - split into words
      2. Lowercase - case normalization
      3. Remove stop words - drop noise words ('the', 'is', 'a')
      4. Stem - reduce words to their root ('engineering' -> 'engin')
    """
    tokens = nltk.word_tokenize(text.lower())
    return [
        STEMMER.stem(tok)
        for tok in tokens
        if tok.isalnum() and tok not in STOP_WORDS
    ]


def generate_bigrams(tokens):
    """
    Generates consecutive word pairs (bi-grams) from a list of tokens.
    Example: ['gpt', '5', 'model'] -> ['gpt_5', '5_model']
    This enables phrase-level matching beyond single keywords.
    """
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def update_bigram_index(cursor, permalink, bigrams):
    """Upserts bigrams for a document into the bigram_index table."""
    for bigram in set(bigrams):  # deduplicate
        cursor.execute("""
            INSERT INTO bigram_index (bigram, permalink)
            VALUES (%s, %s)
            ON CONFLICT (bigram, permalink) DO NOTHING
        """, (bigram, permalink))


def update_inverted_index(cursor, permalink, tokens):
    """
    Given a list of pre-processed tokens for a document, this function:
      1. Computes the Term Frequency (TF) for each unique token.
      2. Upserts each (term, permalink, tf) row into the `inverted_index` table.

    This is the WRITE side of an Inverted Index — the standard data structure
    at the core of every search engine (Lucene, Elasticsearch, Solr, etc.)
    """
    # Count how many times each term appears (Term Frequency)
    tf_map = {}
    for tok in tokens:
        tf_map[tok] = tf_map.get(tok, 0) + 1

    for term, freq in tf_map.items():
        cursor.execute("""
            INSERT INTO inverted_index (term, permalink, term_freq)
            VALUES (%s, %s, %s)
            ON CONFLICT (term, permalink) DO UPDATE
                SET term_freq = EXCLUDED.term_freq
        """, (term, permalink, freq))


def recompute_idf(cursor):
    """
    Recomputes the IDF and average document length in a single ATOMIC transaction.
    Instead of looping in Python, we use a single set-based SQL operation
    for maximum performance and consistency.
    """
    # 1. Total number of documents
    cursor.execute("SELECT COUNT(*) FROM submissions")
    N = cursor.fetchone()[0]
    if N == 0: return

    # 2. Atomic Refresh of term_stats 
    # We use a single query to calculate IDF for ALL terms and upsert them.
    # This is much safer than a Python loop.
    cursor.execute("""
        WITH term_counts AS (
            SELECT term, COUNT(DISTINCT permalink) AS df
            FROM inverted_index
            GROUP BY term
        )
        INSERT INTO term_stats (term, doc_count, idf)
        SELECT 
            term, 
            df,
            LN((%s - df + 0.5) / (df + 0.5) + 1)
        FROM term_counts
        ON CONFLICT (term) DO UPDATE SET
            doc_count = EXCLUDED.doc_count,
            idf       = EXCLUDED.idf;
    """, (N,))

    # 3. Update the global average document length (avgdl)
    cursor.execute("""
        UPDATE system_stats 
        SET stat_value = (SELECT AVG(array_length(tokens, 1)) FROM submissions WHERE tokens IS NOT NULL)
        WHERE stat_name = 'avg_doc_len';
    """)

    print(f"  ACID Update Complete: IDF and avg_doc_len synchronized for {N} documents.")


def update():
    choicesArray = [
        "reddit", "funny", "AskReddit", "gaming", "technology",
        "singularity", "artificial", "machinelearning", "openai",
        "Music", "pics", "science", "worldnews"
    ]

    try:
        conn   = psycopg2.connect(
            dbname=os.getenv('db_name'),
            user=os.getenv('db_user'),
            password=os.getenv('db_password'),
            host='localhost'
        )
        # Use REPEATABLE READ to ensure N (count) doesn't change mid-transaction
        conn.set_session(isolation_level='REPEATABLE READ')
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return

    # --- Step 1: Acquire a Global Advisory Lock (ID: 1010) ---
    # This prevents two instances of update.py from conflicting
    print("Requesting database sync lock...")
    cursor.execute("SELECT pg_advisory_lock(1010);")

    headers = {'User-agent': 'YARP-Engine-Research-Bot-v1.0'}
    new_posts = 0

    try:
        for sub in choicesArray:
            print(f"Fetching r/{sub}...")
            url = f"https://www.reddit.com/r/{sub}/new.json?limit=25"
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"  Skipped (Status: {response.status_code})")
                continue

            data = response.json()
            for post in data['data']['children']:
                item      = post['data']
                permalink = "https://www.reddit.com" + item['permalink']
                title     = item['title']

                tokens  = preprocess(f"{title} {sub}")
                bigrams = generate_bigrams(tokens)

                cursor.execute("""
                    INSERT INTO submissions(title, subreddit, permalink, url, utc, comments, score, tokens, bigrams)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (permalink) DO NOTHING
                """, (
                    title, sub, permalink,
                    item['url'], item['created_utc'],
                    item['num_comments'], item['score'],
                    tokens, bigrams
                ))

                if cursor.rowcount == 1:
                    new_posts += 1
                    update_inverted_index(cursor, permalink, tokens)
                    update_bigram_index(cursor, permalink, bigrams)
            
            print(f"  Processed r/{sub}.")

        # --- Step 2: Recompute IDF inside the SAME transaction ---
        if new_posts > 0:
            print(f"\nIndexing {new_posts} new posts. Finalizing math...")
            recompute_idf(cursor)
        
        # --- Step 3: Atomic Commit ---
        # Everything (posts + index + stats) becomes live at exactly this moment.
        conn.commit()
        print("TRANSACTION COMMITTED: Database synchronized perfectly.")

    except Exception as e:
        print(f"CRITICAL ERROR during update: {e}")
        print("Rolling back transaction to protect database integrity.")
        conn.rollback()
    finally:
        # Release the lock and close connection
        cursor.execute("SELECT pg_advisory_unlock(1010);")
        conn.close()
        print("UPDATE COMPLETED\n")


if __name__ == "__main__":
    print("YARP Engine Active. Press Ctrl+C to stop.")
    while True:
        update()
        print("Waiting 5 minutes for next sync...")
        time.sleep(300)