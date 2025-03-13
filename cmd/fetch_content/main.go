package main

import (
	"database/sql"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"
)

const (
	dbHost     = "localhost"
	dbPort     = 5432
	dbUser     = "user"
	dbPassword = "password"
	dbName     = "github_repo_analysis"
)

// GitHub API Token (Use environment variable for security)
var githubToken = os.Getenv("GITHUB_TOKEN")

// Repository struct to hold database records
type Repository struct {
	ID            int
	NameWithOwner string
	Stars         int
}

func main() {
	host := flag.String("host", "", "database host")
	flag.Parse()

	if *host == "" {
		*host = dbHost
	}

	// Connect to PostgreSQL
	psqlInfo := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		*host, dbPort, dbUser, dbPassword, dbName,
	)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Fetch repositories from DB
	rows, err := db.Query("SELECT id, name_with_owner FROM repositories")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	i := 0
	for rows.Next() {
		if i == 10 {
			break
		}
		var repo Repository
		if err := rows.Scan(&repo.ID, &repo.NameWithOwner); err != nil {
			log.Fatal(err)
		}

		// Split name_with_owner into owner and repo name
		values := strings.Split(repo.NameWithOwner, "/")
		owner := values[0]
		repoName := values[1]

		// Fetch README content
		readme, err := fetchReadme(owner, repoName)
		if err != nil {
			logrus.Fatal(err)
		}

		// Store fetched content
		storeReadme(db, repo.ID, readme)
		i++
	}

	fmt.Println("Data fetch and store completed.")
}

// Fetch README from GitHub API
func fetchReadme(owner, repo string) (string, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/%s/readme", owner, repo)
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "token "+githubToken)
	req.Header.Set("Accept", "application/vnd.github.v3.raw")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		logrus.Error(resp.StatusCode)
		logrus.Error("response error: ", string(body))
		return "", fmt.Errorf("README not found")
	}

	body, _ := io.ReadAll(resp.Body)
	return string(body), nil
}

func storeReadme(db *sql.DB, repoID int, readme string) {
	// Insert or update data
	query := `
	INSERT INTO repo_docs (repo_id, readme)
	VALUES ($1, $2)
	ON CONFLICT (repo_id) DO UPDATE SET readme = EXCLUDED.readme`
	_, err := db.Exec(query, repoID, readme)
	if err != nil {
		log.Printf("Failed to store data for repo %d: %v", repoID, err)
	} else {
		fmt.Printf("Stored README for repo %d\n", repoID)
	}
}
