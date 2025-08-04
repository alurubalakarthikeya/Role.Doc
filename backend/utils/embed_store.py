import os
import re
import json
import math
import textwrap
import faiss
from collections import Counter

VECTOR_DIR = "vector_db"
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)


class SimpleVectorizer:
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}

    def _tokenize(self, text):
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        return [word for word in words if word not in stop_words and len(word) > 2]

    def fit_transform(self, documents):
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        word_freq = Counter()
        for tokens in tokenized_docs:
            word_freq.update(set(tokens))

        most_common = word_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

        num_docs = len(documents)
        for word in self.vocabulary:
            doc_count = sum(1 for tokens in tokenized_docs if word in tokens)
            self.idf_values[word] = math.log(num_docs / (doc_count + 1))

        vectors = []
        for tokens in tokenized_docs:
            vector = [0.0] * len(self.vocabulary)
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            for word, count in token_counts.items():
                if word in self.vocabulary:
                    tf = count / total_tokens
                    idf = self.idf_values[word]
                    vector[self.vocabulary[word]] = tf * idf
            vectors.append(vector)
        return vectors

    def transform(self, documents):
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            vector = [0.0] * len(self.vocabulary)
            token_counts = Counter(tokens)
            total_tokens = len(tokens) if tokens else 1
            for word, count in token_counts.items():
                if word in self.vocabulary:
                    tf = count / total_tokens
                    idf = self.idf_values.get(word, 0)
                    vector[self.vocabulary[word]] = tf * idf
            vectors.append(vector)
        return vectors


def embed_and_store(text, file_name):
    section_pattern = r"(?:\n|^)(\d[\d\.]*[\)\.]?|Step \d+|Section \d+)[^\n]*\n"
    sections = re.split(section_pattern, text)

    structured_chunks = []
    i = 0
    while i < len(sections):
        if i < len(sections) and re.match(section_pattern, sections[i]):
            heading = sections[i].strip()
            if i + 1 < len(sections):
                content = sections[i + 1].strip()
                full_text = f"{heading}: {content}"
                chunks = textwrap.wrap(full_text, width=450, break_long_words=False, break_on_hyphens=False)
                structured_chunks.extend(chunks)
                i += 2
            else:
                i += 1
        else:
            if i < len(sections):
                plain = sections[i].strip()
                if plain:
                    chunks = textwrap.wrap(plain, width=450, break_long_words=False, break_on_hyphens=False)
                    structured_chunks.extend(chunks)
            i += 1

    structured_chunks = [chunk for chunk in structured_chunks if chunk.strip()]
    if not structured_chunks:
        print(f"Warning: No chunks extracted from {file_name}")
        return

    vectorizer = SimpleVectorizer(max_features=300)
    vectors = vectorizer.fit_transform(structured_chunks)


    import numpy as np
    index = faiss.IndexFlatL2(len(vectors[0])) 
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, os.path.join(VECTOR_DIR, f"{file_name}.index"))
    with open(os.path.join(VECTOR_DIR, f"{file_name}_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(structured_chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(VECTOR_DIR, f"{file_name}_vectorizer.json"), "w", encoding="utf-8") as f:
        json.dump({
            "vocab": vectorizer.vocabulary,
            "idf": vectorizer.idf_values
        }, f, indent=2)


def load_vectorizer(file_name):
    path = os.path.join(VECTOR_DIR, f"{file_name}_vectorizer.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vectorizer = SimpleVectorizer()
    vectorizer.vocabulary = {k: int(v) for k, v in data["vocab"].items()}
    vectorizer.idf_values = data["idf"]
    return vectorizer


def query_vector_store(query, file_name, top_k=3):
    index_path = os.path.join(VECTOR_DIR, f"{file_name}.index")
    chunks_path = os.path.join(VECTOR_DIR, f"{file_name}_chunks.json")
    vectorizer_path = os.path.join(VECTOR_DIR, f"{file_name}_vectorizer.json")

    if not all(os.path.exists(p) for p in [index_path, chunks_path, vectorizer_path]):
        return "Document not indexed."

    try:
        index = faiss.read_index(index_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        vectorizer = load_vectorizer(file_name)
        query_vec = vectorizer.transform([query])[0]

        import numpy as np
        query_array = np.array([query_vec], dtype="float32")
        distances, indices = index.search(query_array, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(chunks):
                results.append(chunks[idx])
        return "\n\n".join(results) if results else "No relevant chunks found."

    except Exception as e:
        return f"Error during query: {str(e)}"
