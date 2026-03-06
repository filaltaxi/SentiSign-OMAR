# slm/src/generate_sentence.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts a sign-language word buffer into a grammatically correct sentence.
#
# Sign language buffers are:
#   - ALL CAPS (from recognition model output)
#   - Missing articles (a, the), prepositions (to, at, in),
#     auxiliary verbs (will, is, am, have), conjunctions
#   - Often in topic-comment order, not English order
#
# Example:
#   Input : ["I", "TOMORROW", "HOSPITAL", "GO"]
#   Output: "I will go to the hospital tomorrow."
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sentence_model import generate

SYSTEM_PROMPT = (
    "You convert short ASL-style word buffers into one natural English sentence. "
    "Use every input word. Add only the minimum grammar words needed. "
    "Keep the meaning faithful. Do not explain your reasoning. "
    "Output exactly one sentence and nothing else."
)

STRICT_SYSTEM_PROMPT = (
    "You repair ASL-style word buffers into one short English sentence. "
    "Do not add new content words, events, objects, descriptions, or reasons. "
    "Only add grammar helper words such as articles, prepositions, auxiliaries, and pronouns. "
    "Output exactly one sentence and nothing else."
)

ALLOWED_HELPER_WORDS = {
    "a", "an", "the", "to", "at", "in", "on", "of", "for", "from", "with", "into", "by",
    "and", "but", "or", "if", "because", "that", "this", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "not", "no", "yes", "please",
}

SUBJECT_WORDS = {"i", "you", "we", "mother", "father", "child", "family"}
TIME_WORDS = {"today", "tomorrow", "now"}
PLACE_WORDS = {"hospital", "home", "toilet"}
STATE_PHRASES = {
    "hungry": "hungry",
    "sick": "sick",
    "pain": "in pain",
    "sorry": "sorry",
}
ACTION_WORDS = {"go", "come", "need", "want", "help", "sleep", "understand"}
ARTICLE_OBJECTS = {
    "doctor": "a doctor",
    "hospital": "the hospital",
    "toilet": "the toilet",
}


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _word_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 2:
        variants.update({
            token + "s",
            token + "es",
            token + "ed",
            token + "ing",
            token.rstrip("e") + "ing",
        })
    return {variant for variant in variants if variant}


def _unexpected_content_words(result: str, clean_words: list[str]) -> set[str]:
    allowed = set(ALLOWED_HELPER_WORDS)
    for word in clean_words:
        for token in _tokenise(word):
            allowed.update(_word_variants(token))

    return {
        token for token in _tokenise(result)
        if token not in allowed and len(token) > 1
    }


def _clean_model_text(result: str) -> str:
    result = re.sub(r"<think>.*?</think>", " ", result, flags=re.IGNORECASE | re.DOTALL)
    result = result.strip()
    if "Words:" in result:
        result = result.split("Words:")[-1].strip()
    if "Sentence:" in result:
        result = result.split("Sentence:")[-1].strip()
    result = re.sub(r"^(answer|output|assistant)\s*:\s*", "", result, flags=re.IGNORECASE)
    result = result.strip().strip('"').strip("'").strip()
    if "\n" in result:
        result = next((line.strip() for line in result.splitlines() if line.strip()), result)
    return result


def _choose_subject(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in SUBJECT_WORDS:
            return token
    return None


def _subject_phrase(subject: str | None) -> str:
    if not subject:
        return "I"
    return subject.capitalize()


def _be_verb(subject: str | None) -> str:
    if subject in {"you", "we"}:
        return "are"
    if subject == "i" or subject is None:
        return "am"
    return "is"


def _main_verb(subject: str | None, infinitive: str) -> str:
    if subject in {None, "i", "you", "we"}:
        return infinitive
    if infinitive.endswith("y") and infinitive[-2:] != "ay":
        return infinitive[:-1] + "ies"
    if infinitive.endswith(("s", "sh", "ch", "x", "z", "o")):
        return infinitive + "es"
    return infinitive + "s"


def _object_phrase(token: str) -> str:
    return ARTICLE_OBJECTS.get(token, token)


def _rule_based_fallback(clean_words: list[str]) -> str:
    tokens = [token for word in clean_words for token in _tokenise(word)]
    subject = _choose_subject(tokens)
    subject_text = _subject_phrase(subject)
    time_tokens = [token for token in tokens if token in TIME_WORDS]
    time_suffix = f" {time_tokens[0]}" if time_tokens else ""

    if "thank" in tokens and "you" in tokens:
        if "help" in tokens:
            return "Thank you for your help."
        return "Thank you."

    if "sorry" in tokens:
        return f"{subject_text} {_be_verb(subject)} sorry."

    state_tokens = [token for token in tokens if token in STATE_PHRASES]
    state_clause = None
    if state_tokens:
        state_clause = f"{subject_text} {_be_verb(subject)} {STATE_PHRASES[state_tokens[0]]}"

    content_objects = [
        token for token in tokens
        if token not in SUBJECT_WORDS
        and token not in TIME_WORDS
        and token not in STATE_PHRASES
        and token not in ACTION_WORDS
        and token not in {"thank", "yes", "no", "not", "what", "where"}
    ]
    object_text = _object_phrase(content_objects[0]) if content_objects else None

    if "go" in tokens or "come" in tokens:
        movement = "come" if "come" in tokens else "go"
        place = next((token for token in content_objects if token in PLACE_WORDS), None)
        destination = _object_phrase(place) if place else None
        if destination:
            return f"{subject_text} will {movement} to {destination}{time_suffix}."
        return f"{subject_text} will {movement}{time_suffix}."

    if "need" in tokens:
        clause = f"{subject_text} {_main_verb(subject, 'need')}"
        if object_text:
            clause = f"{clause} {object_text}"
        if state_clause:
            return f"{state_clause} and {_main_verb(subject, 'need')} {object_text or 'help'}{time_suffix}."
        return f"{clause}{time_suffix}."

    if "want" in tokens:
        clause = f"{subject_text} {_main_verb(subject, 'want')}"
        if object_text:
            clause = f"{clause} {object_text}"
        if state_clause:
            return f"{state_clause} and {_main_verb(subject, 'want')} {object_text or 'help'}{time_suffix}."
        return f"{clause}{time_suffix}."

    if "help" in tokens:
        return f"{subject_text} {_main_verb(subject, 'need')} help{time_suffix}."

    if state_clause:
        return f"{state_clause}{time_suffix}."

    return " ".join(clean_words) + "."


def words_to_sentence(words: list) -> str:
    """
    Convert a list of sign-language words into a natural English sentence.

    Args:
        words: e.g. ["I", "TOMORROW", "HOSPITAL", "GO"]

    Returns:
        A clean, grammatically correct English sentence.
    """
    if not words:
        raise ValueError("[generate_sentence] Word list is empty.")

    # Normalise: strip whitespace, title-case, drop empty tokens
    clean = [w.strip().capitalize() for w in words if w.strip()]
    if not clean:
        raise ValueError("[generate_sentence] No valid words after cleaning.")

    word_string = ", ".join(clean)

    # Smaller local models behave better with explicit rules and compact examples.
    prompt = (
        "Rules:\n"
        "- Use all input words.\n"
        "- Add missing grammar words only when needed.\n"
        "- Keep the sentence short, natural, and grammatically correct.\n"
        "- Output one sentence only.\n\n"
        "Examples:\n"
        "Words: I, Tomorrow, Hospital, Go\n"
        "Sentence: I will go to the hospital tomorrow.\n\n"
        "Words: Hungry, I, Food, Want\n"
        "Sentence: I am hungry and want food.\n\n"
        f"Words: {word_string}\n"
        "Sentence:"
    )

    print(f"[generate_sentence] Input : {words}")
    result = _clean_model_text(generate(prompt, max_length=80, system_prompt=SYSTEM_PROMPT))

    # ── Post-processing ───────────────────────────────────────────────────────
    extras = _unexpected_content_words(result, clean)
    if extras:
        print(f"[generate_sentence] Extra content detected {sorted(extras)} — retrying with stricter prompt.")
        strict_prompt = (
            "Rules:\n"
            "- Use all input words.\n"
            "- Do not add any new content words.\n"
            "- You may only add helper grammar words.\n"
            "- Output one short sentence only.\n\n"
            f"Words: {word_string}\n"
            "Sentence:"
        )
        retry = _clean_model_text(
            generate(strict_prompt, max_length=32, system_prompt=STRICT_SYSTEM_PROMPT)
        )
        retry_extras = _unexpected_content_words(retry, clean)
        if retry and not retry_extras:
            result = retry
        elif retry_extras:
            print(
                f"[generate_sentence] Retry still added extra content {sorted(retry_extras)} "
                "- using rule-based fallback."
            )
            result = _rule_based_fallback(clean)

    # Fallback if model returns garbage
    if not result or len(result.split()) < 2:
        print("[generate_sentence] Weak output — using rule-based fallback.")
        result = _rule_based_fallback(clean)

    # Ensure ends with punctuation
    if result[-1] not in ".!?":
        result += "."

    # Capitalise first character
    result = result[0].upper() + result[1:]

    print(f"[generate_sentence] Output: {result}")
    return result


# Backwards-compatible alias expected by `main.py`.
def generate_sentence(words: list) -> str:
    return words_to_sentence(words)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ["I", "TOMORROW", "HOSPITAL", "GO"],
        ["HUNGRY", "I", "FOOD", "WANT"],
        ["MOTHER", "SICK", "DOCTOR", "NEED"],
        ["THANK", "YOU", "HELP"],
        ["WATER", "NEED", "I"],
    ]
    print("\n" + "=" * 56)
    print("  Sentence Model  Sign → Sentence Test")
    print("=" * 56)
    for words in tests:
        sentence = words_to_sentence(words)
        print(f"  IN : {words}")
        print(f"  OUT: {sentence}\n")
