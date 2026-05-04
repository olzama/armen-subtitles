"""Language code/name normalization utilities.

Accepts ISO 639-2 three-letter codes or full English language names (case-insensitive).
Always returns the canonical full name for use in file paths and JSON keys.
"""

# ISO 639-2 code -> canonical full name
LANG_MAP = {
    "afr": "Afrikaans",
    "ara": "Arabic",
    "bel": "Belarusian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Chinese",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fas": "Persian",
    "fin": "Finnish",
    "fra": "French",
    "glg": "Galician",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jpn": "Japanese",
    "kat": "Georgian",
    "kor": "Korean",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mkd": "Macedonian",
    "msa": "Malay",
    "nld": "Dutch",
    "nor": "Norwegian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "vie": "Vietnamese",
    "zho": "Chinese",
}

# Reverse map: lowercase full name -> canonical full name
_NAME_MAP = {name.lower(): name for name in LANG_MAP.values()}

# Reverse map: canonical full name -> ISO code (last code wins for duplicates like zho/cmn)
_NAME_TO_CODE = {name: code for code, name in LANG_MAP.items()}


def normalize_lang(value: str) -> str:
    """Return the canonical full language name for a code or name.

    Accepts ISO 639-2 codes (e.g. 'rus', 'ENG') or full names (e.g. 'Russian', 'english').
    Raises ValueError with a helpful message if the input is not recognized.
    """
    v = value.strip()

    # Try as ISO code
    canonical = LANG_MAP.get(v.lower())
    if canonical:
        return canonical

    # Try as full name
    canonical = _NAME_MAP.get(v.lower())
    if canonical:
        return canonical

    codes = ", ".join(sorted(LANG_MAP))
    names = ", ".join(sorted(_NAME_MAP.values()))
    raise ValueError(
        f"Unrecognized language '{value}'.\n"
        f"  Accepted ISO 639-2 codes: {codes}\n"
        f"  Accepted names: {names}"
    )


def lang_code(value: str) -> str:
    """Return the ISO 639-2 code for a code or full language name.

    Accepts the same inputs as normalize_lang but returns the three-letter code
    rather than the full name.  Use this for JSON keys and directory names.
    """
    v = value.strip()

    # Already a valid code?
    if v.lower() in LANG_MAP:
        return v.lower()

    # Try as full name
    canonical = _NAME_MAP.get(v.lower())
    if canonical:
        return _NAME_TO_CODE[canonical]

    codes = ", ".join(sorted(LANG_MAP))
    names = ", ".join(sorted(_NAME_MAP.values()))
    raise ValueError(
        f"Unrecognized language '{value}'.\n"
        f"  Accepted ISO 639-2 codes: {codes}\n"
        f"  Accepted names: {names}"
    )
