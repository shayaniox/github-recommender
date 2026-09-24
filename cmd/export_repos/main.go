package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
)

// Repo struct with Topics field
type Repo struct {
	Owner  string  `json:"owner"`
	Name   string  `json:"name"`
	Topics []topic `json:"topics"`
}

type topic struct {
	Name  string `json:"name"`
	Stars int    `json:"stars"`
}

// Reads all topics from the CSV file and returns a set for fast lookup
func loadAllowedTopics(csvPath string) (map[string]struct{}, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open topics CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	allowed := make(map[string]struct{})

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV header: %w", err)
	}
	topicIdx := -1
	for i, col := range header {
		if strings.TrimSpace(col) == "topic" {
			topicIdx = i
			break
		}
	}
	if topicIdx == -1 {
		return nil, fmt.Errorf("no 'topic' column found in CSV")
	}

	// Read topics
	for {
		record, err := reader.Read()
		if err != nil {
			break // EOF is expected
		}
		topic := strings.TrimSpace(record[topicIdx])
		if topic != "" {
			allowed[topic] = struct{}{}
		}
	}
	return allowed, nil
}

// Checks if any topic in repoTopics exists in allowedTopics
func hasAllowedTopic(repoTopics []topic, allowedTopics map[string]struct{}) bool {
	for _, t := range repoTopics {
		if _, ok := allowedTopics[t.Name]; ok {
			return true
		}
	}
	return false
}

func main() {
	// Load allowed topics from CSV
	allowedTopics, err := loadAllowedTopics("all_topics.csv")
	if err != nil {
		log.Fatalf("Error loading topics: %v", err)
	}

	// Open the JSON file for reading
	jsonFile, err := os.Open("dataset/pretty-dataset.json")
	if err != nil {
		log.Fatalf("Failed to open JSON file: %v", err)
	}
	defer jsonFile.Close()

	// Open the output file for writing
	outFile, err := os.Create("repos_list.txt")
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	dec := json.NewDecoder(jsonFile)

	// Read the opening bracket of the array
	t, err := dec.Token()
	if err != nil {
		log.Fatalf("Failed to read opening token: %v", err)
	}
	if delim, ok := t.(json.Delim); !ok || delim != '[' {
		log.Fatalf("Expected opening '[' for JSON array")
	}

	// Iterate through the array elements
	for dec.More() {
		var repo Repo
		if err := dec.Decode(&repo); err != nil {
			log.Fatalf("Failed to decode repo: %v", err)
		}
		if hasAllowedTopic(repo.Topics, allowedTopics) && repo.Owner != "" && repo.Name != "" {
			_, err := fmt.Fprintf(outFile, "%s/%s\n", repo.Owner, repo.Name)
			if err != nil {
				log.Fatalf("Failed to write to output file: %v", err)
			}
		}
	}

	// Read the closing bracket of the array
	t, err = dec.Token()
	if err != nil {
		log.Fatalf("Failed to read closing token: %v", err)
	}
	if delim, ok := t.(json.Delim); !ok || delim != ']' {
		log.Fatalf("Expected closing ']' for JSON array")
	}

	fmt.Println("Extraction complete. Output written to repos_list.txt")
}
