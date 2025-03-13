FROM golang:1.22-bookworm

# Set the working directory inside the container
WORKDIR /app

# Copy the Go source code
COPY ./cmd/ ./cmd/
COPY go.mod .
COPY go.sum .

# Download Go dependencies
RUN go mod tidy

# Build the Go binary
RUN go build ./cmd/convert_dataset/
RUN go build ./cmd/fetch_content/
