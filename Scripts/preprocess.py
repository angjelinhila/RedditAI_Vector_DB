import json
import os
import re

INPUT_FILE = "data/raw/reddit_data.jsonl"
OUTPUT_FILE = "data/processed/reddit_cleaned.jsonl"

def clean_text(text):
    """Basic text cleaning: remove URLs, special characters, and extra spaces."""
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def preprocess_reddit_data():
    """Read raw Reddit data, clean it, and save the processed output."""
    with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            data["body"] = clean_text(data["body"])  # Clean post body
            json.dump(data, outfile)
            outfile.write("\n")

if __name__ == "__main__":
    preprocess_reddit_data()
    print("Preprocessing complete! Processed data saved in", OUTPUT_FILE)


import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer

nltk.download("punkt")
nltk.download("stopwords")

# Directories
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load ColBERT tokenizer (for FAISS processing)
colbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Stopwords for optional removal
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    """Basic text cleaning: remove URLs, special characters, and extra spaces."""
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

### 1️⃣ Preprocessing for Database (BM25 + FAISS) ###
def preprocess_for_db(input_file, output_file):
    """Preprocesses data for BM25 (Elasticsearch) and FAISS (ColBERT)."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            
            # Clean text
            data["body"] = clean_text(data["body"])
            
            # FAISS-specific: tokenize for ColBERT
            data["tokenized_body"] = colbert_tokenizer.tokenize(data["body"])
            
            json.dump(data, outfile)
            outfile.write("\n")
    print(f"Database preprocessing complete: {output_file}")

### 2️⃣ Preprocessing for Modelling (Deep Learning) ###
def preprocess_for_modelling(input_file, output_file):
    """Prepares text for machine learning: stopword removal, lemmatization, and tokenization."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            
            # Clean text
            text = clean_text(data["body"])
            
            # Sentence tokenization
            sentences = sent_tokenize(text)
            
            # Word tokenization with stopword removal
            tokens = [word.lower() for word in word_tokenize(text) if word.lower() not in STOPWORDS]
            
            # Store processed text
            data["sentences"] = sentences
            data["tokens"] = tokens
            
            json.dump(data, outfile)
            outfile.write("\n")
    print(f"Modelling preprocessing complete: {output_file}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_for_db(RAW_DATA_DIR + "reddit_data.jsonl", PROCESSED_DATA_DIR + "reddit_cleaned_db.jsonl")
    preprocess_for_modelling(RAW_DATA_DIR + "reddit_data.jsonl", PROCESSED_DATA_DIR + "reddit_cleaned_model.jsonl")
