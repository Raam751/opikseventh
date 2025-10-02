import os
import json
import re
import math
from collections import defaultdict, Counter
from typing import List

# -----------------------------
# Utility Functions
# -----------------------------

def tokenize(text: str) -> List[str]:
    """Split text into lowercase words, removing non-alphabetic chars."""
    return re.findall(r'\b[a-z]+\b', text.lower())

def compute_tf(tokens: List[str]) -> dict:
    """Compute term frequency for each word in a document."""
    tf = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}

def compute_idf(documents: List[List[str]]) -> dict:
    """Compute inverse document frequency for all words."""
    N = len(documents)
    idf = {}
    all_tokens = set([word for doc in documents for word in set(doc)])
    for word in all_tokens:
        containing = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((N + 1) / (containing + 1)) + 1
    return idf

def compute_tfidf(tokens: List[str], idf: dict) -> dict:
    """Compute TF-IDF for a document."""
    tf = compute_tf(tokens)
    return {word: tf[word] * idf.get(word, 0) for word in tf}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[w] * vec2[w] for w in common)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0.0

# -----------------------------
# Knowledge Base
# -----------------------------

class KnowledgeBase:
    def __init__(self, storage_file="knowledge.json"):
        self.storage_file = storage_file
        self.entries = []
        self.load()

    def add_entry(self, title: str, content: str):
        """Add a new knowledge entry."""
        self.entries.append({"title": title, "content": content})
        self.save()

    def save(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.entries, f, indent=4)

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def search(self, query: str, top_k=3):
        """Search knowledge base using TF-IDF and cosine similarity."""
        if not self.entries:
            return []

        documents = [tokenize(e["content"]) for e in self.entries]
        idf = compute_idf(documents)

        query_tokens = tokenize(query)
        query_vec = compute_tfidf(query_tokens, idf)

        scores = []
        for i, tokens in enumerate(documents):
            doc_vec = compute_tfidf(tokens, idf)
            score = cosine_similarity(query_vec, doc_vec)
            scores.append((score, self.entries[i]))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]

# -----------------------------
# Command Line Interface
# -----------------------------

def main():
    kb = KnowledgeBase()

    print("=== Personal Knowledge Assistant ===")
    print("Commands: add, search, list, exit")
    print("-----------------------------------")

    while True:
        cmd = input("\nEnter command: ").strip().lower()

        if cmd == "add":
            title = input("Title: ")
            content = input("Content: ")
            kb.add_entry(title, content)
            print("Entry added!")

        elif cmd == "list":
            if not kb.entries:
                print("No entries yet.")
            else:
                for i, e in enumerate(kb.entries, 1):
                    print(f"{i}. {e['title']}")

        elif cmd == "search":
            query = input("Search query: ")
            results = kb.search(query)
            if not results:
                print("No results found.")
            else:
                for score, entry in results:
                    print(f"\n📌 {entry['title']} (score={score:.2f})")
                    print(entry['content'])

        elif cmd == "exit":
            print("Goodbye!")
            break

        else:
            print("Unknown command. Try add/search/list/exit.")

if __name__ == "__main__":
    main()
