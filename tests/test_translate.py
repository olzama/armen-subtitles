import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from translate import get_chunks


SAMPLE_SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nFirst subtitle block.\n\n"
    "2\n00:00:02,500 --> 00:00:03,500\nSecond subtitle block.\n\n"
    "3\n00:00:04,000 --> 00:00:05,000\nThird subtitle block."
)


def test_single_chunk_when_under_limit():
    chunks = get_chunks(SAMPLE_SRT, max_chars=5000)
    assert len(chunks) == 1


def test_splits_into_multiple_chunks():
    # Each block is ~60 chars; max_chars=70 forces one block per chunk
    chunks = get_chunks(SAMPLE_SRT, max_chars=70)
    assert len(chunks) == 3


def test_chunk_content_integrity():
    chunks = get_chunks(SAMPLE_SRT, max_chars=70)
    assert len(chunks) == 3
    assert "First subtitle block" in chunks[0]
    assert "Second subtitle block" in chunks[1]
    assert "Third subtitle block" in chunks[2]
