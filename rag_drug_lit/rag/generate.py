"""Takes the passages we retrieved and asks Claude to answer from them.

The key idea: we tell Claude to use ONLY the passages we give it and to
cite which document each fact came from. That keeps answers grounded and
makes them traceable.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

INSTRUCTIONS = """Act as a drug-discovery research assistant. Answer the
question using ONLY the passages provided. Each passage starts with its source
in square brackets, like [ad_dupilumab.txt]. After each fact, cite the source it
came from in square brackets. If the passages do not contain the answer, state
that clearly instead of guessing."""


def build_prompt(question, passages):
    """Stitch the retrieved passages and the question into one prompt string."""
    lines = []
    for p in passages:
        lines.append("[" + p["source"] + "] " + p["text"])
    context = "\n\n".join(lines)
    return "Passages:\n\n" + context + "\n\nQuestion: " + question


def answer(question, passages, model="claude-haiku-4-5"):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_prompt(question, passages)

    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
