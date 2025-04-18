# GitHub Repository Metadata Extraction and Analysis

This project fetches, processes, and analyzes GitHub repository metadata and README files from a large dataset. The extracted data is stored in **PostgreSQL**, and later, **TF-IDF** is applied to analyze README content.

---

## 📌 Prerequisites
Ensure you have the following installed before running the project:

| Dependency  | Version  | Installation Guide |
|------------|---------|--------------------|
| **Go**     | 1.22+   | [Install Go](https://go.dev/doc/install) |
| **PostgreSQL** | 14+ | [Install PostgreSQL](https://www.postgresql.org/download/) |
| **Docker**  | 24.0+  | [Install Docker](https://docs.docker.com/get-docker/) |
| **Docker Compose** | 2.20+ | [Install Docker Compose](https://docs.docker.com/compose/install/) |
| **Kaggle API** | Latest | [Install Kaggle API](https://www.kaggle.com/docs/api) |

---

## 📂 Project Structure
```
📦 github-repo-analysis
├── cmd
│   ├── convert_dataset
│   │   └── main.go
│   └── fetch_content
│       └── main.go
├── docker-compose.yml
├── Dockerfile
├── go.mod
├── go.sum
├── LICENSE
├── README.md
└── scripts
```

---

## 📥 Downloading the Dataset
Before running the project, download the dataset from **Kaggle**
```bash
#!/bin/bash

mkdir -p dataset # make directory if not exists

curl -L -o ./dataset/dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/pelmers/github-repository-metadata-with-5-stars
```
Then, extract the dataset
```bash
unzip ~/dataset/dataset.zip -d dataset/
```

This is a minified JSON file, for structuring the JSON content:
```bash
cd dataset
jq '.' repo_metadata.json formatted.json
```
---

## 📦 Setting Up the Database

### 1️⃣ Create PostgreSQL Schema
Run the following SQL commands to set up the required tables

```sql
-- Table to store repository metadata
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    name_with_owner TEXT UNIQUE NOT NULL,
    stars INTEGER NOT NULL
);

-- Table to store topics of repositories
CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    stars INTEGER NOT NULL
);

-- Table to store README files linked to repositories
CREATE TABLE repo_docs (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER UNIQUE REFERENCES repositories(id) ON DELETE CASCADE,
    readme TEXT
);

CREATE TABLE repo_docs_preprocessed (
    repo_id INTEGER PRIMARY KEY REFERENCES repositories(id) ON DELETE CASCADE,
    processed_readme TEXT NOT NULL
);

CREATE TABLE repo_docs_tfidf (
    repo_id INTEGER PRIMARY KEY REFERENCES repositories(id) ON DELETE CASCADE,
    tfidf_vector JSONB NOT NULL
);
```

### 2️⃣ Running Database with Docker
To start PostgreSQL in **Docker**, run:
```bash
docker-compose up -d
```
This will start the database and the program inside a **Docker network**.

---

## 🐳 Docker Setup

### 📌 Dockerfile for the Go Application
Create a `Dockerfile` in the project root

```dockerfile
# Use the official Go image as base
FROM golang:1.22

# Set the working directory inside the container
WORKDIR /app

# Copy the Go source code
COPY ./cmd/ .

# Download Go dependencies
RUN go mod tidy

# Build the Go binary
RUN go build ./cmd/convert_dataset/
RUN go build ./cmd/fetch_content/
```

---

## 📌 Docker-Compose Configuration
To run both **PostgreSQL** and the **Go application** in the same **network**, create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  database:
    image: postgres:14
    container_name: github_repo_db
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: github_repo_analysis
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  pgdata:
```

---

## 🚀 Running the Project

### 1️⃣ Start Everything with Docker
```bash
docker-compose up -d
```

### 2️⃣ Run the Go Application Manually
If running outside Docker

+ converting dataset to database records
```bash
go run ./cmd/convert_dataset/main.go
```
+ fetching README content of repositories
```bash
go run ./cmd/fetch_content/main.go
```

Using Docker:

```bash
export DATASET=./dataset/dataset.json # <your dataset file>
docker run -it --rm --network=github-recommender_default \
  -v ./dataset/:/app/dataset/ \
  -e DB_HOST=database \
  -e DB_PORT=5432 \
  -e DB_USER=user \
  -e DB_PASSWORD=password \
  -e DB_NAME=github_recommender \
  github_recommender ./convert_dataset -dataset $DATASET -host github_repo_db
```

> [!NOTE]
> If running the app outside the docker, you don't need to provide `host` option, since the default host is `localhost`.

### 3️⃣ Check PostgreSQL Data
After running, check the database with:
```sql
SELECT * FROM repositories;
SELECT * FROM repo_docs;
```

---
## Analyzing README Files

After storing repository metadata and README content, the next steps involve processing the README files to extract meaningful information using NLP techniques.

### 1. Preprocess README Content

```bash
python src/preprocess.py
```

### 2. Calculate the TF-IDF

```bash
python src/tfidf.py
```

