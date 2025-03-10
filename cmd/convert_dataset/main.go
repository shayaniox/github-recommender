package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	_ "github.com/lib/pq"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// Repository struct for JSON decoding
type Repository struct {
	NameWithOwner string  `json:"nameWithOwner"`
	Stars         int     `json:"stars"`
	Topics        []Topic `json:"topics"`
}

// Topic struct (linked to Repository)
type Topic struct {
	Name     string `json:"name"`
	Stars    int    `json:"stars"`
	RepoAddr string `json:"-"`
}

// PostgreSQL connection details
const (
	host     = "localhost"
	port     = 5432
	user     = "github-recommender"
	password = "github-recommender"
	dbname   = "github-recommender"
)

// Insert repositories in batch
func insertRepositories(db *sql.DB, repos []Repository) (map[string]int, error) {
	if len(repos) == 0 {
		return nil, nil
	}

	var values []string
	var params []interface{}
	paramCount := 1

	for _, repo := range repos {
		values = append(values, fmt.Sprintf("($%d, $%d)", paramCount, paramCount+1))
		params = append(params, repo.NameWithOwner, repo.Stars)
		paramCount += 2
	}

	// FIX: Remove RETURNING for batch insert
	query := fmt.Sprintf(
		"INSERT INTO repositories (name_with_owner, stars) VALUES %s ON CONFLICT (name_with_owner) DO NOTHING",
		strings.Join(values, ","),
	)

	_, err := db.Exec(query, params...)
	if err != nil {
		return nil, err
	}

	// Fetch inserted repo IDs
	repoIDMap := make(map[string]int)
	rows, err := db.Query(
		"SELECT id, name_with_owner FROM repositories WHERE name_with_owner = ANY(ARRAY[name_with_owner])",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var name string
		if err := rows.Scan(&id, &name); err != nil {
			return nil, err
		}
		repoIDMap[name] = id
	}
	return repoIDMap, nil
}

// Insert topics in batch
func insertTopics(db *sql.DB, topics []Topic, repoIDMap map[string]int) error {
	if len(topics) == 0 {
		return nil
	}

	var values []string
	var params []interface{}
	paramCount := 1

	for _, topic := range topics {
		repoID, exists := repoIDMap[topic.RepoAddr]
		if !exists {
			continue
		}
		values = append(values, fmt.Sprintf("($%d, $%d, $%d)", paramCount, paramCount+1, paramCount+2))
		params = append(params, repoID, topic.Name, topic.Stars)
		paramCount += 3
	}

	query := fmt.Sprintf(
		"INSERT INTO topics (repo_id, name, stars) VALUES %s",
		strings.Join(values, ","),
	)
	_, err := db.Exec(query, params...)
	return errors.Wrap(err, "error on db exec")
}

// Stream JSON and insert data in batches
func processJSON(db *sql.DB, jsonFile string, batchSize int) error {
	file, err := os.Open(jsonFile)
	if err != nil {
		return errors.Wrap(err, "error on open file")
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	_, err = decoder.Token() // Read opening '['
	if err != nil {
		return errors.Wrap(err, "decoder token")
	}

	var repoBatch []Repository
	var topicBatch []Topic

	for decoder.More() {
		var repo Repository
		if err := decoder.Decode(&repo); err != nil {
			log.Fatal(err)
		}

		repoBatch = append(repoBatch, repo)
		topicBatch = append(topicBatch, repo.Topics...)
		for i := range topicBatch {
			topicBatch[i].RepoAddr = repo.NameWithOwner
		}

		if len(repoBatch) >= batchSize {
			repoIDMap, err := insertRepositories(db, repoBatch)
			if err != nil {
				return errors.Wrap(err, "error on insert repositories")
			}

			if err := insertTopics(db, topicBatch, repoIDMap); err != nil {
				return errors.Wrap(err, "insert topics")
			}
			repoBatch, topicBatch = []Repository{}, []Topic{} // Reset batches
		}
	}

	// Insert remaining data
	if len(repoBatch) > 0 {
		repoIDMap, err := insertRepositories(db, repoBatch)
		if err != nil {
			return errors.Wrap(err, "error on insert repositories")
		}
		if err := insertTopics(db, topicBatch, repoIDMap); err != nil {
			return errors.Wrap(err, "insert topics")
		}
	}

	fmt.Println("✅ Data successfully stored in PostgreSQL.")
	return nil
}

func main() {
	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable", host, port, user, password, dbname)
	logrus.Info(psqlInfo)
	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	log.SetFlags(log.Lshortfile | log.Ldate)

	jsonFile := "dataset/pretty-dataset.json" // Update with actual file
	batchSize := 1000                         // Adjust batch size for performance
	if err := processJSON(db, jsonFile, batchSize); err != nil {
		log.Fatal(err)
	}
}
