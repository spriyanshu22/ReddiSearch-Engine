import os
import psycopg2
from flask import Flask, render_template, request
from dotenv import load_dotenv

# --- NLP Preprocessing (query-side only) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

load_dotenv()
app = Flask(__name__)

STEMMER    = SnowballStemmer("english")
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):
    """
    Identical pipeline to update.py — CRITICAL that both sides are identical.
    The query terms must be stemmed the same way as the stored index terms,
    otherwise 'engineering' would never match the stored token 'engin'.
    """
    tokens = nltk.word_tokenize(text.lower())
    return [
        STEMMER.stem(tok)
        for tok in tokens
        if tok.isalnum() and tok not in STOP_WORDS
    ]


def generate_bigrams(tokens):
    """Generates consecutive token pairs for phrase-level matching."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv('db_name'),
        user=os.getenv('db_user'),
        password=os.getenv('db_password')
    )


@app.route('/')
def index():
    user_query = request.args.get('q', '').strip()

    # AI Lens keywords — always applied as an AND filter
    ai_lens    = "openai gpt claude agentic llm deepseek neural"
    ai_tokens  = preprocess(ai_lens)

    conn = get_db_connection()
    cur  = conn.cursor()

    if user_query:
        query_tokens = preprocess(user_query)

        if not query_tokens:
            # If search is only stop words, show default AI news
            cur.execute("""
                SELECT  title, subreddit, permalink, score
                FROM    submissions
                WHERE   tokens && %s::text[]
                ORDER   BY utc DESC
                LIMIT   100
            """, (ai_tokens,))
        else:
            # --- SEMANTIC EXPANSION ---
            cur.execute("SELECT term_b FROM semantic_synonyms WHERE term_a = ANY(%s)", (query_tokens,))
            synonyms       = [row[0] for row in cur.fetchall()]
            expanded_tokens = list(set(query_tokens + synonyms))
            if synonyms:
                print(f"Semantically expanded: {query_tokens} -> {expanded_tokens}")

            # --- BI-GRAM QUERY TERMS ---
            # If user typed more than one word, generate bigrams from query
            query_bigrams = generate_bigrams(query_tokens) if len(query_tokens) > 1 else []

            # --- HYBRID BM25 + BI-GRAM SEARCH ---
            # Unigram BM25 scoring (primary relevance signal)
            # UNION with bigram exact-phrase matches (given a score bonus)
            cur.execute("""
                WITH
                -- Unigram BM25 scoring
                matched_postings AS (
                    SELECT  ii.permalink,
                            ii.term_freq                                AS tf,
                            ts.idf,
                            array_length(s.tokens, 1)                   AS doc_len,
                            (SELECT stat_value FROM system_stats
                             WHERE stat_name = 'avg_doc_len')           AS avgdl
                    FROM    inverted_index  ii
                    JOIN    term_stats      ts  ON ts.term     = ii.term
                    JOIN    submissions     s   ON s.permalink = ii.permalink
                    WHERE   ii.term = ANY(%s)
                    AND     s.tokens && %s::text[]
                ),
                bm25_scores AS (
                    SELECT  permalink,
                            SUM(
                                idf * (tf * 2.5)
                                / (tf + 1.5 * (0.25 + 0.75 * doc_len / avgdl))
                            ) AS bm25_score
                    FROM    matched_postings
                    GROUP BY permalink
                ),

                -- Bi-gram phrase bonus: posts with exact phrases get a +2.0 bonus
                bigram_bonus AS (
                    SELECT  permalink,
                            2.0 AS bonus
                    FROM    bigram_index
                    WHERE   bigram = ANY(%s)
                    GROUP BY permalink
                ),

                -- Combine both scores
                final_scores AS (
                    SELECT  COALESCE(b.permalink, bg.permalink) AS permalink,
                            COALESCE(b.bm25_score, 0) + COALESCE(bg.bonus, 0) AS total_score
                    FROM    bm25_scores b
                    FULL OUTER JOIN bigram_bonus bg ON bg.permalink = b.permalink
                )

                SELECT  s.title, s.subreddit, s.permalink, s.score
                FROM    final_scores  f
                JOIN    submissions   s ON s.permalink = f.permalink
                ORDER   BY f.total_score DESC
                LIMIT   100
            """, (expanded_tokens, ai_tokens, query_bigrams if query_bigrams else ['__no_bigrams__']))

        results = cur.fetchall()

        # --- QUERY LOGGING ---
        # Store every search in the DB for analytics
        if query_tokens:
            cur.execute("""
                INSERT INTO query_logs (query, result_count)
                VALUES (%s, %s)
            """, (user_query.lower().strip(), len(results)))
            conn.commit()

    else:
        # Default view: most recent AI news
        cur.execute("""
            SELECT  title, subreddit, permalink, score
            FROM    submissions
            WHERE   tokens && %s::text[]
            ORDER   BY utc DESC
            LIMIT   100
        """, (ai_tokens,))
        results = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('index.html', results=results, query=user_query)


@app.route('/analytics')
def analytics():
    """
    Dashboard showing search analytics:
      - Top 10 most searched terms
      - Total searches made
      - Recent search history
    """
    conn = get_db_connection()
    cur  = conn.cursor()

    # Top 10 most searched queries
    cur.execute("""
        SELECT  query,
                COUNT(*)        AS search_count,
                AVG(result_count)::INT AS avg_results
        FROM    query_logs
        GROUP BY query
        ORDER BY search_count DESC
        LIMIT   10
    """)
    top_queries = cur.fetchall()

    # Total searches ever made
    cur.execute("SELECT COUNT(*) FROM query_logs")
    total_searches = cur.fetchone()[0]

    # Recent search history (last 20)
    cur.execute("""
        SELECT query, result_count, searched_at
        FROM   query_logs
        ORDER  BY searched_at DESC
        LIMIT  20
    """)
    recent = cur.fetchall()

    # Total posts indexed
    cur.execute("SELECT COUNT(*) FROM submissions")
    total_posts = cur.fetchone()[0]

    # Total unique terms in inverted index
    cur.execute("SELECT COUNT(*) FROM term_stats")
    total_terms = cur.fetchone()[0]

    cur.close()
    conn.close()

    return render_template('analytics.html',
                           top_queries=top_queries,
                           total_searches=total_searches,
                           recent=recent,
                           total_posts=total_posts,
                           total_terms=total_terms)


if __name__ == '__main__':
    app.run(debug=True)