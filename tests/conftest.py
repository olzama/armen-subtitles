import sys
import types
import pytest


@pytest.fixture(scope="session", autouse=True)
def stub_api_modules():
    """Stub out openai and google.genai so modules can be imported without API keys."""
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = object
        sys.modules["openai"] = openai_stub

    if "google" not in sys.modules:
        google_stub = types.ModuleType("google")
        genai_stub = types.ModuleType("google.genai")
        types_stub = types.ModuleType("google.genai.types")
        genai_stub.types = types_stub
        google_stub.genai = genai_stub
        sys.modules["google"] = google_stub
        sys.modules["google.genai"] = genai_stub
        sys.modules["google.genai.types"] = types_stub


@pytest.fixture
def two_item_translation_data():
    return {
        "title": "MINIMAL TEST",
        "model": "gpt-5.2",
        "items": [
            {
                "id": 1,
                "character": "Bunsha",
                "original": {"rus": "Меня опять терзают смутные сомнения…"},
                "reference": {"eng": "Vague doubts haunt me once again..."},
                "analysis": "Mock-elevated theatrical tone matters.",
                "segment_number": [569],
                "translations": {
                    "eng": {
                        "zero":       {"1": "I'm tormented by vague doubts again…",
                                       "2": "Vague doubts plague me again..."},
                        "characters": {"1": "Once again I'm tormented by vague misgivings...",
                                       "2": "I'm plagued by vague doubts again..."},
                    }
                },
            },
            {
                "id": 2,
                "character": "Ivan the Terrible",
                "original": {"rus": "Оставь меня, старушка, я в печали"},
                "reference": {"eng": "Leave me, old woman, I am in sorrow."},
                "analysis": "Archaic elevated register matters.",
                "segment_number": [597],
                "translations": {
                    "eng": {
                        "zero":       {"1": "Leave me, old woman, I'm in sorrow.",
                                       "2": "Leave me alone, old woman, I'm sad."},
                        "characters": {"1": "Leave me, old woman, I am in sorrow.",
                                       "2": "Go away, old woman, I'm grieving."},
                    }
                },
            },
        ],
    }
