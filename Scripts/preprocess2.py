import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Directories
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load ColBERT tokenizer (for FAISS processing)
colbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Stopwords for optional removal
STOPWORDS = set(stopwords.words("english"))

# Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Basic text cleaning: remove URLs, special characters, and extra spaces."""
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def chunk_text(text, tokenizer, max_length=512, stride=256):
    """Splits text into overlapping chunks to fit within model limits."""
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i: i + max_length] for i in range(0, len(tokens), stride)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

### 1️⃣ Preprocessing for Database (BM25 + FAISS) ###
def preprocess_for_db():
    """Preprocesses data for BM25 (Elasticsearch) and FAISS (ColBERT)."""
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith("_comments.json") or filename.endswith("_submissions.json"):
            input_file = os.path.join(RAW_DATA_DIR, filename)
            output_file = os.path.join(PROCESSED_DATA_DIR, filename.replace(".json", "_cleaned.json"))
            
            with open(input_file, "r") as infile, open(output_file, "w") as outfile:
                for line in infile:
                    data = json.loads(line)
                    
                    # Process body or selftext depending on file type
                    text_field = "selftext" if "selftext" in data else "body"
                    data[text_field] = clean_text(data.get(text_field, ""))
                    
                    # FAISS-specific: Chunk tokenized text for ColBERT
                    data["tokenized_body"] = chunk_text(data[text_field], colbert_tokenizer)
                    
                    json.dump(data, outfile)
                    outfile.write("\n")
            print(f"Database preprocessing complete: {output_file}")

### 2️⃣ Preprocessing for Modelling (Deep Learning) ###
def preprocess_for_modelling():
    """Prepares text for machine learning: stopword removal, lemmatization, and tokenization."""
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith("_comments.json") or filename.endswith("_submissions.json"):
            input_file = os.path.join(RAW_DATA_DIR, filename)
            output_file = os.path.join(PROCESSED_DATA_DIR, filename.replace(".json", "_cleaned_model.json"))
            
            with open(input_file, "r") as infile, open(output_file, "w") as outfile:
                for line in infile:
                    data = json.loads(line)
                    
                    # Process body or selftext depending on file type
                    text_field = "selftext" if "selftext" in data else "body"
                    text = clean_text(data.get(text_field, ""))
                    
                    # Sentence tokenization
                    sentences = sent_tokenize(text)
                    
                    # Word tokenization with stopword removal and lemmatization
                    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) if word.lower() not in STOPWORDS]
                    
                    # Store processed text
                    data["sentences"] = sentences
                    data["tokens"] = tokens
                    
                    json.dump(data, outfile)
                    outfile.write("\n")
            print(f"Modelling preprocessing complete: {output_file}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_for_db()
    preprocess_for_modelling()
