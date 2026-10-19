#!/usr/bin/env python3
"""Download the Russian source subtitles listed in sources.csv and verify their checksums.

The source subtitle files are not redistributed. Each row of sources.csv gives
the OpenSubtitles.org file ID and the SHA-256 of the exact file used in the
paper; segment numbers in the dataset refer to that file. Requires an
OpenSubtitles API key (https://www.opensubtitles.com/consumers).

Usage:
    OPENSUBTITLES_API_KEY=... python code/fetch_sources.py [--csv sources.csv] [--out data]
"""

import argparse
import csv
import hashlib
import os
import sys
import urllib.request
import json
from pathlib import Path

API = "https://api.opensubtitles.com/api/v1/download"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("sources.csv"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    key = os.environ.get("OPENSUBTITLES_API_KEY")
    if not key:
        sys.exit("Set OPENSUBTITLES_API_KEY.")
    ok = True
    for row in csv.DictReader(open(a.csv)):
        dst = a.out / row["film"] / "subs" / row["file"]
        if not row["opensubtitles_id"]:
            print(f"{row['film']}: no OpenSubtitles ID recorded"); ok = False; continue
        req = urllib.request.Request(API, data=json.dumps({"file_id": int(row["opensubtitles_id"])}).encode(),
                                     headers={"Api-Key": key, "Content-Type": "application/json",
                                              "User-Agent": "soviet-meme-mt v1"})
        link = json.load(urllib.request.urlopen(req))["link"]
        data = urllib.request.urlopen(link).read()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        match = digest == row["sha256"]
        ok &= match
        print(f"{row['film']}: {'ok' if match else 'CHECKSUM MISMATCH'} ({dst})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
