import re
import psycopg2
import nltk

# Ensure required resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

# Download necessary resources (only run once)
# nltk.download("stopwords")
nltk.download("punkt_tab")

# Initialize stop words and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Cleans and preprocesses a README text."""
    print("Original text:", text[:100] + "..." if len(text) > 100 else text)

    text = text.lower()  # Convert to lowercase
    print("After lowercase:", text[:100] + "..." if len(text) > 100 else text)

    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    print("After removing punctuation:", text[:100] + "..." if len(text) > 100 else text)

    tokens = word_tokenize(text)  # Tokenize
    print("After tokenization:", tokens[:20], "..." if len(tokens) > 20 else "")

    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    print("After removing stopwords:", tokens[:20], "..." if len(tokens) > 20 else "")

    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    print("After stemming:", tokens[:20], "..." if len(tokens) > 20 else "")

    result = " ".join(tokens)  # Convert back to text
    print("Final result:", result[:100] + "..." if len(result) > 100 else result)

    return result


# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="github_repo_analysis",
    user="user",
    password="password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Fetch README files
cursor.execute("SELECT repo_id, readme FROM repo_docs WHERE readme IS NOT NULL")
readme_files = cursor.fetchall()

# Process and update database
for repo_id, readme in readme_files:
    processed_text = preprocess_text(readme)  # Call the function with readme as argument
    query = """
    INSERT INTO repo_docs_preprocessed (repo_id, processed_readme)
    VALUES (%s, %s)
    ON CONFLICT (repo_id) DO UPDATE SET processed_readme = EXCLUDED.processed_readme;
    """
    cursor.execute(query, (repo_id, processed_text))
    break

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()

print("Done pre-processing ✅")
