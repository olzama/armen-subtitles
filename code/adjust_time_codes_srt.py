import re
import sys

import pysrt

# --- Tuning knobs ---
WPM = 180  # words per minute reading speed (150–220 are common)
MIN_SHORT_SEC = 4.0
AVG_SEC = 6.0
LONG_SEC = 8.0

SHORT_WORDS_MAX = 3      # "a couple of words"
LONG_WORDS_MIN = 18      # "long" threshold

MAX_OVERLAP_MS = 500     # allowed overlap into next subtitle start
ALLOW_SHORTEN = False    # set True if you also want to reduce overly-long durations
MIN_DURATION_MS = 1000   # hard floor so nothing becomes absurdly short

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def count_words(text: str) -> int:
    # Join multi-line subtitles; strip tags lightly
    text = re.sub(r"<[^>]+>", "", text)  # remove simple HTML tags if present
    return len(WORD_RE.findall(text))


def desired_duration_seconds(word_count: int) -> float:
    """
    Rule requested:
      - couple words => >=4s
      - average => 6s
      - long => 8s
    We also incorporate reading-time estimation as a "smart" baseline.
    """
    # reading time baseline
    seconds_per_word = 60.0 / WPM
    reading_time = word_count * seconds_per_word

    if word_count <= SHORT_WORDS_MAX:
        # ensure short lines aren't flashed too fast
        return max(MIN_SHORT_SEC, reading_time)
    elif word_count >= LONG_WORDS_MIN:
        # long lines get more time, up to ~8s
        return min(LONG_SEC, max(AVG_SEC, reading_time))
    else:
        # average: center around 6s, but don't undercut reading time
        return max(AVG_SEC, reading_time)


def to_ms(t: pysrt.SubRipTime) -> int:
    return t.ordinal


def from_ms(ms: int) -> pysrt.SubRipTime:
    return pysrt.SubRipTime(milliseconds=max(int(ms), 0))


def adjust_timings(subs: pysrt.SubRipFile) -> pysrt.SubRipFile:
    for i in range(len(subs)):
        cur = subs[i]
        start_ms = to_ms(cur.start)
        end_ms = to_ms(cur.end)
        cur_dur_ms = max(end_ms - start_ms, 0)

        text = cur.text or ""
        wc = count_words(text)
        target_sec = desired_duration_seconds(wc)
        target_ms = int(target_sec * 1000)

        # Hard safety floor
        target_ms = max(target_ms, MIN_DURATION_MS)

        # Decide whether we will change this subtitle duration
        if (not ALLOW_SHORTEN) and (cur_dur_ms >= target_ms):
            continue

        proposed_end = start_ms + target_ms

        if i < len(subs) - 1:
            next_start = to_ms(subs[i + 1].start)
            # Don't go too far into next subtitle (allow some overlap)
            max_end = next_start + MAX_OVERLAP_MS
            new_end = min(proposed_end, max_end)
        else:
            new_end = proposed_end

        if new_end != end_ms:
            # Ensure end is not before start
            new_end = max(new_end, start_ms + MIN_DURATION_MS)
            cur.end = from_ms(new_end)

    return subs


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    subs = pysrt.open(input_path, encoding="utf-8")
    subs = adjust_timings(subs)
    subs.save(output_path, encoding="utf-8")

    print(f"Saved adjusted SRT to: {output_path}")