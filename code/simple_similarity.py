import sys
import json
import numpy as np
import pysrt
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import unicodedata

def is_punctuation(token):
    return all(unicodedata.category(ch).startswith("P") for ch in token)

nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=False)

def tokenize(text):
    doc = nlp(normalize_text(text))
    tokens = []
    for sent in doc.sentences:
        for token in sent.tokens:
            tokens.append(token.text.lower())
    return tokens

def split_words_punct(tokens):
    words, punct = [], []
    for t in tokens:
        if is_punctuation(t):
            punct.append(t)
        else:
            words.append(t)
    return words, punct

def extract_text_from_srt(path):
    subs = pysrt.open(path, encoding="utf-8")
    texts = [sub.text.replace("\n", " ").strip() for sub in subs]
    return "\n".join(texts)


def chunk_text(text, max_words=800):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
    return chunks

def shingles(seq, k):
    if len(seq) < k:
        return set()
    return {tuple(seq[i:i+k]) for i in range(len(seq) - k + 1)}

def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

def containment(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    return inter / min(len(a_set), len(b_set))


def surface_similarity_weighted(text_a, text_b, word_k=3, punct_k=3, alpha=0.1):
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    words_a, punct_a = split_words_punct(tokens_a)
    words_b, punct_b = split_words_punct(tokens_b)

    score_x_punct = jaccard(shingles(words_a, word_k), shingles(words_b, word_k))
    score_punct = jaccard(shingles(tokens_a, punct_k), shingles(tokens_b, punct_k))

    return {
        "surface": (1 - alpha) * score_x_punct + alpha * score_punct,
        "score_x_punct": score_x_punct,
        "score_punct": score_punct,
        "alpha": alpha,
        "word_k": word_k,
        "punct_k": punct_k,
    }

def normalize_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())   # collapse whitespace
    text = text.lower()
    return text

def chunk_surface_diagnostics(text_a, text_b,
                                       chunk_size=500,
                                       word_k=3,
                                       punct_k=3,
                                       alpha=0.1,
                                       top_n=5):

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    # Pre-split B into chunks
    b_chunks = [
        tokens_b[i:i+chunk_size]
        for i in range(0, len(tokens_b), chunk_size)
    ]

    # Precompute B chunk shingles
    b_chunk_shingles = []
    for chunk in b_chunks:
        words_b, _ = split_words_punct(chunk)
        b_chunk_shingles.append({
            "word": shingles(words_b, word_k),
            "token": shingles(chunk, punct_k)
        })

    out = []

    for i in range(0, len(tokens_a), chunk_size):
        chunk_a = tokens_a[i:i+chunk_size]
        words_a, _ = split_words_punct(chunk_a)

        sh_words_a = shingles(words_a, word_k)
        sh_tokens_a = shingles(chunk_a, punct_k)

        best_score = 0.0

        for b_sh in b_chunk_shingles:
            w = jaccard(sh_words_a, b_sh["word"])
            p = jaccard(sh_tokens_a, b_sh["token"])
            s = (1 - alpha) * w + alpha * p
            if s > best_score:
                best_score = s

        preview = " ".join(chunk_a)[:200]

        out.append({
            "start_token": i,
            "end_token": i + len(chunk_a),
            "surface_similarity": best_score,
            "preview": preview
        })

    out.sort(key=lambda x: x["surface_similarity"])
    return out[:top_n]


def lexical_similarity(text_a, text_b):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000
    )
    tfidf = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def chunked_semantic_similarity(chunks_a, chunks_b, model):
    emb_a = model.encode(chunks_a, normalize_embeddings=True)
    emb_b = model.encode(chunks_b, normalize_embeddings=True)

    sim_matrix = np.matmul(emb_a, emb_b.T)

    # For each chunk in A, find best match in B
    score_a = sim_matrix.max(axis=1).mean()

    # For each chunk in B, find best match in A
    score_b = sim_matrix.max(axis=0).mean()

    return float((score_a + score_b) / 2.0)


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_srt.py file1.srt file2.srt [output.json]")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "similarity_results.json"

    print("Reading SRT files...")
    text_a = extract_text_from_srt(file_a)
    text_b = extract_text_from_srt(file_b)

    print("Computing surface similarity...")
    surf_score = surface_similarity_weighted(text_a, text_b, word_k=3, punct_k=3, alpha=0.1)

    print("Chunking text for semantic similarity...")
    chunks_a = chunk_text(text_a)
    chunks_b = chunk_text(text_b)

    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Computing semantic similarity...")
    sem_score = chunked_semantic_similarity(chunks_a, chunks_b, model)

    balanced_score = 0.6 * surf_score["surface"] + 0.4 * sem_score

    results = {
        "file_a": file_a,
        "file_b": file_b,
        "surface_similarity": surf_score,
        "semantic_similarity": sem_score,
        "balanced_similarity": balanced_score
    }

    print("\nResults:")
    print(json.dumps(results, indent=4))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nRunning chunk-level surface diagnostics...")
    diagnostics = chunk_surface_diagnostics(
        text_a,
        text_b,
        chunk_size=500,
        word_k=3, punct_k=3, alpha=0.1, top_n=5
    )

    print("\nMost divergent chunks:")
    for d in diagnostics:
        print("\n---")
        print(f"Similarity: {d['surface_similarity']:.4f}")
        print(f"Token span: {d['start_token']}–{d['end_token']}")
        print(f"Preview: {d['preview']}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
