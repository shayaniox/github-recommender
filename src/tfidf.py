import psycopg2
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Database connection
def get_db_connection():
    return psycopg2.connect(
        dbname="github_repo_analysis",
        user="user",
        password="password",
        host="localhost",
        port="5432"
    )

# Fetch preprocessed README content
def fetch_preprocessed_readmes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT repo_id, processed_readme FROM repo_docs_preprocessed;")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data  # Returns list of (repo_id, processed_readme)

# Compute TF-IDF and store results
def compute_tfidf():
    data = fetch_preprocessed_readmes()

    repo_ids = [row[0] for row in data]
    documents = [row[1] for row in data]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    # Convert to a list of dictionaries (repo_id -> {word: score})
    tfidf_data = [
        (repo_ids[i], json.dumps(dict(zip(feature_names, tfidf_matrix[i].toarray()[0]))))
        for i in range(len(repo_ids))
    ]

    # Store in DB
    store_tfidf(tfidf_data)

# Store TF-IDF vectors in the database
def store_tfidf(tfidf_data):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO repo_docs_tfidf (repo_id, tfidf_vector)
    VALUES (%s, %s)
    ON CONFLICT (repo_id) DO UPDATE SET tfidf_vector = EXCLUDED.tfidf_vector;
    """

    cursor.executemany(query, tfidf_data)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    compute_tfidf()
