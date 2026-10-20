#!/usr/bin/env python3
"""Download the Russian source subtitles from OpenSubtitles and rebuild the exact files used in the paper.

The source subtitle files are not redistributed. sources.json gives, for each
film, the OpenSubtitles file ID and a small recipe that turns the downloaded
file into the file the paper's segment numbers refer to: byte-order mark,
line endings, an entry-number offset, and a few edits to blank lines. The
script checks the SHA-256 of both the downloaded and the rebuilt file.

Downloading requires an OpenSubtitles account and API key
(https://www.opensubtitles.com/consumers); the free tier allows 20 downloads a day.

Usage:
    OPENSUBTITLES_API_KEY=... OPENSUBTITLES_USERNAME=... OPENSUBTITLES_PASSWORD=... \\
        python code/fetch_sources.py [--sources sources.json] [--out data]
Writes <out>/<film>/subs/<film>-rus.srt.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

API = "https://api.opensubtitles.com/api/v1/"


def lines(data):
    return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def apply_recipe(data, recipe):
    """The downloaded file (bytes) -> the file used in the paper (bytes)."""
    ls = lines(data)
    off = recipe.get("renumber_offset", 0)
    if off:
        ls = [str(int(l) + off) if re.fullmatch(r"\d+", l) and k + 1 < len(ls) and "-->" in ls[k + 1] else l
              for k, l in enumerate(ls)]
    for i1, i2, new in sorted(recipe.get("line_edits", []), reverse=True):
        ls[i1:i2] = new
    text = ("\r\n" if recipe.get("crlf") else "\n").join(ls)
    return ("﻿" if recipe.get("bom") else "").encode("utf-8") + text.encode("utf-8")


def call(path, body, headers):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=Path("sources.json"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    env = {k: os.environ.get(f"OPENSUBTITLES_{k.upper()}") for k in ("api_key", "username", "password")}
    if not all(env.values()):
        sys.exit("Set OPENSUBTITLES_API_KEY, OPENSUBTITLES_USERNAME and OPENSUBTITLES_PASSWORD.")
    headers = {"Api-Key": env["api_key"], "Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "soviet-meme-mt v1"}
    headers["Authorization"] = "Bearer " + call("login", {"username": env["username"], "password": env["password"]},
                                                headers)["token"]
    ok = True
    for src in json.loads(a.sources.read_text(encoding="utf-8")):
        link = call("download", {"file_id": src["opensubtitles_file_id"]}, headers)["link"]
        raw = urllib.request.urlopen(urllib.request.Request(link, headers={"User-Agent": headers["User-Agent"]}),
                                     timeout=60).read()
        if hashlib.sha256(raw).hexdigest() != src["download_sha256"]:
            print(f"{src['film']}: the downloaded file has changed on OpenSubtitles (checksum mismatch)")
            ok = False
            continue
        data = apply_recipe(raw, src["recipe"])
        match = hashlib.sha256(data).hexdigest() == src["sha256"]
        ok &= match
        dst = a.out / src["film"] / "subs" / src["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"{src['film']}: {'ok' if match else 'REBUILT FILE DIFFERS'} ({dst})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
