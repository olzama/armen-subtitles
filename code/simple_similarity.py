import sys
import json
import numpy as np
import pysrt
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import unicodedata
import difflib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def save_matrix_csv(matrix, files, output_path):
    labels = [os.path.basename(f) for f in files]
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(output_path+"/similarity_matrix.csv")

def get_srt_files(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(".srt"):
            files.append(os.path.join(folder, f))
    files.sort()
    return files


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


def surface_similarity_weighted(text_a,
                                text_b,
                                alpha=0.1):
    """
    Surface similarity based on token-level LCS.

    alpha: weight given to full-token channel (punctuation included).
           word channel weight = (1 - alpha)
    """

    # Tokenize (your tokenize() already normalizes)
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    # Separate word-only tokens
    words_a, _ = split_words_punct(tokens_a)
    words_b, _ = split_words_punct(tokens_b)

    # --- Word-level LCS ---
    matcher_words = difflib.SequenceMatcher(None, words_a, words_b)
    lcs_words = sum(block.size for block in matcher_words.get_matching_blocks())
    score_words = lcs_words / min(len(words_a), len(words_b)) if words_a and words_b else 0.0

    # --- Full-token LCS (includes punctuation) ---
    matcher_tokens = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    lcs_tokens = sum(block.size for block in matcher_tokens.get_matching_blocks())
    score_tokens = lcs_tokens / min(len(tokens_a), len(tokens_b)) if tokens_a and tokens_b else 0.0

    surface = (1 - alpha) * score_words + alpha * score_tokens

    return {
        "surface": surface,
        "score_words_lcs": score_words,
        "score_tokens_lcs": score_tokens,
        "alpha": alpha
    }

def normalize_text(text):
    replacements = {
        # Apostrophes
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‛": "'",

        # Quotes (including Spanish guillemets)
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',

        # Dashes
        "–": "-",
        "—": "-",
        "―": "-",

        # Ellipsis
        "…": "...",

        # Spanish inverted punctuation
        "¿": "",  # remove inverted question mark
        "¡": "",  # remove inverted exclamation mark
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.lower()
    text = " ".join(text.split())
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

def compute_similarity_matrix(folder):
    files = get_srt_files(folder)
    n = len(files)

    if n < 2:
        raise ValueError("Need at least two .srt files in folder.")

    texts = [extract_text_from_srt(f) for f in files]

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            score = surface_similarity_weighted(
                texts[i],
                texts[j]
            )["surface"]

            matrix[i, j] = score
            matrix[j, i] = score

    return matrix, files

def plot_similarity_matrix(matrix, files, output_dir, filename="similarity_matrix.png"):
    labels = [os.path.basename(f) for f in files]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="magma",
        vmin=0,
        vmax=1
    )

    plt.title("LCS Surface Similarity Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_folder.py folder_path")
        sys.exit(1)

    folder = sys.argv[1]
    output_dir = sys.argv[2]

    print("Computing similarity matrix...")
    matrix, files = compute_similarity_matrix(folder)

    print("Matrix:")
    print(matrix)

    save_matrix_csv(matrix, files, output_dir)
    plot_similarity_matrix(matrix, files, output_dir)



def run_diagnostics(text_a, text_b):
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


if __name__ == "__main__":
    main()
