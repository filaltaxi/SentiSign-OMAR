#!/usr/bin/env python3
"""
Sentence-only desktop pipeline (no TTS).

Flow:
1) Opens webcam (sign + emotion capture)
2) Press ENTER (or press Enter/q in the window) to stop capture
3) Generates and prints the sentence (no audio)

Run with uv:
  uv run python run_pipeline_sentence_only.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
_SLM_SRC = os.path.join(_ROOT, "slm", "src")

for p in [_SRC, _SLM_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_sentence import words_to_sentence
from sign_recognizer import capture_words_and_emotion


def run():
    words, emotion = capture_words_and_emotion()

    print("\n" + "─" * 64)
    print("  Generating sentence ...")
    sentence = words_to_sentence(words)

    print("\n" + "═" * 64)
    print("  ✓  Sentence generated")
    print(f"  Words   : {words}")
    print(f"  Emotion : {emotion}")
    print(f'  Sentence: "{sentence}"')
    print("═" * 64)


if __name__ == "__main__":
    run()

