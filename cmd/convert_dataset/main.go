package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// Repository struct for JSON decoding
type Repository struct {
	Name   string  `json:"name"`
	Owner  string  `json:"owner"`
	Stars  int     `json:"stars"`
	Forks  int     `json:"forks"`
	Issues int     `json:"issues"`
	Topics []Topic `json:"topics"`
}

// Topic struct (linked to Repository)
type Topic struct {
	Name string `json:"name"`
}

var allowedTopics map[string]struct{}
var csvPath = "all_topics.csv"

func init() {
	var err error
	allowedTopics, err = loadAllowedTopics(csvPath)
	if err != nil {
		logrus.Fatal(err)
	}
}

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
func hasAllowedTopic(repoTopics []Topic, allowedTopics map[string]struct{}) bool {
	for _, t := range repoTopics {
		if _, ok := allowedTopics[t.Name]; ok {
			return true
		}
	}
	return false
}

func main() {
	dataset := flag.String("dataset", "", "Path to the dataset file")
	outputFile := flag.String("output", "repositories.csv", "Path to the output CSV file")
	flag.Parse()

	if *dataset == "" {
		fmt.Println("Error: Dataset path is required. Use -dataset flag.")
		os.Exit(1)
	}

	log.SetFlags(log.Lshortfile | log.Ldate)

	jsonFile := *dataset
	batchSize := 100 // Adjust batch size for performance
	if err := processJSON(jsonFile, *outputFile, batchSize); err != nil {
		log.Fatal(err)
	}
}

// Write repositories to CSV
func writeRepositoriesToCSV(csvWriter *csv.Writer, repos []Repository) error {
	if len(repos) == 0 {
		return nil
	}

	for _, repo := range repos {
		// Convert topics to a comma-separated string
		topicNames := make([]string, len(repo.Topics))
		for i, topic := range repo.Topics {
			topicNames[i] = topic.Name
		}

		nameWithOwner := fmt.Sprintf("%s/%s", repo.Owner, repo.Name)

		// Write repository data to CSV
		record := []string{
			nameWithOwner,
			strconv.Itoa(repo.Stars),
			strconv.Itoa(repo.Forks),
			strconv.Itoa(repo.Issues),
			strings.Join(topicNames, ","),
		}

		if err := csvWriter.Write(record); err != nil {
			return errors.Wrap(err, "error writing record to csv")
		}
	}

	return csvWriter.Error()
}

// Stream JSON and write data to CSV in batches
func processJSON(jsonFile string, outputFile string, batchSize int) error {
	file, err := os.Open(jsonFile)
	if err != nil {
		return errors.Wrap(err, "error on open file")
	}
	defer file.Close()

	// Create CSV file
	csvFile, err := os.Create(outputFile)
	if err != nil {
		return errors.Wrap(err, "error creating CSV file")
	}
	defer csvFile.Close()

	// Initialize CSV writer
	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// Write CSV header
	// header := []string{"NameWithOwner", "Stars", "Forks", "Issues", "Topics"}
	// if err := csvWriter.Write(header); err != nil {
	// 	return errors.Wrap(err, "error writing CSV header")
	// }

	decoder := json.NewDecoder(file)
	_, err = decoder.Token() // Read opening '['
	if err != nil {
		return errors.Wrap(err, "decoder token")
	}

	var repoBatch []Repository

	for decoder.More() {
		var repo Repository
		if err := decoder.Decode(&repo); err != nil {
			log.Fatal(err)
		}

		if hasAllowedTopic(repo.Topics, allowedTopics) && repo.Owner != "" && repo.Name != "" {
			repoBatch = append(repoBatch, repo)

			if len(repoBatch) >= batchSize {
				if err := writeRepositoriesToCSV(csvWriter, repoBatch); err != nil {
					return errors.Wrap(err, "error writing repositories to CSV")
				}
				repoBatch = []Repository{} // Reset batch
				csvWriter.Flush()          // Flush to file
				// break
			}
		}
	}

	// Write remaining data
	if len(repoBatch) > 0 {
		if err := writeRepositoriesToCSV(csvWriter, repoBatch); err != nil {
			return errors.Wrap(err, "error writing repositories to CSV")
		}
	}

	fmt.Println("✅ Data successfully stored in CSV file:", outputFile)
	return nil
}
