package main

import (
	"database/sql"
	"encoding/json"
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
	dbUser     = "github-recommender"
	dbPassword = "github-recommender"
	dbName     = "github-recommender"
)

// GitHub API Token (Use environment variable for security)
var githubToken = os.Getenv("GITHUB_TOKEN")

// Repository struct to hold database records
type Repository struct {
	ID            int
	NameWithOwner string
	Stars         int
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

// Fetch Wiki Home Page (if available)
func fetchWiki(owner, repo string) (string, error) {
	url := fmt.Sprintf("https://raw.githubusercontent.com/wiki/%s/%s/Home.md", owner, repo)
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		logrus.Error(resp.StatusCode)
		logrus.Error("response error: ", string(body))
		return "", fmt.Errorf("wiki not found")
	}

	body, _ := io.ReadAll(resp.Body)
	return string(body), nil
}

func checkWikiPages(owner, repo string) error {
	// GitHub API URL for the repository's wiki
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s", owner, repo)

	// Make a GET request to the GitHub API
	resp, err := http.Get(apiURL)
	if err != nil {
		return fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// Check if the request was successful
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch repository info: status code %d", resp.StatusCode)
	}

	// Decode the JSON response
	var repoInfo struct {
		HasWiki bool   `json:"has_wiki"`
		WikiURL string `json:"html_url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&repoInfo); err != nil {
		return fmt.Errorf("failed to decode JSON response: %v", err)
	}

	logrus.WithFields(logrus.Fields{
		"has wiki": repoInfo.HasWiki,
		"wiki url": repoInfo.WikiURL,
	}).Info()

	return nil
}

func main() {
	// Connect to PostgreSQL
	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		dbHost, dbPort, dbUser, dbPassword, dbName)

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

		// Fetch README and Wiki content
		readme, err := fetchReadme(owner, repoName)
		if err != nil {
			logrus.Fatal(err)
		}
		checkWikiPages(owner, repoName)

		// Store fetched content
		storeReadmeAndWiki(db, repo.ID, readme)
		i++
	}

	fmt.Println("Data fetch and store completed.")
}

func storeReadmeAndWiki(db *sql.DB, repoID int, readme string) {
	// Insert or update data
	query := `
	INSERT INTO repo_docs (repo_id, readme, wiki)
	VALUES ($1, $2, $3)
	ON CONFLICT (repo_id) DO UPDATE SET readme = EXCLUDED.readme, wiki = EXCLUDED.wiki`
	_, err := db.Exec(query, repoID, readme, "")
	if err != nil {
		log.Printf("Failed to store data for repo %d: %v", repoID, err)
	} else {
		fmt.Printf("Stored README & Wiki for repo %d\n", repoID)
	}
}
