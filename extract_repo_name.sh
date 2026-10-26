#!/bin/bash

JSON_FILE="dataset/pretty-dataset.json"
# JSON_FILE="dataset/sample-dataset.json"
OUTPUT_FILE="repos_list.txt"
jq -r '.[] | "\(.owner)/\(.name)"' "$JSON_FILE" > "$OUTPUT_FILE"

echo "Repository list has been written to $OUTPUT_FILE"