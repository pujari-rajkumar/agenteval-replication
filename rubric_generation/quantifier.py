"""Checkpoint 3: Score conversations against the generated rubric."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from .autogen_compat import run_assistant_sync

QUANTIFIER_SYSTEM_MESSAGE = (
    "You are an expert evaluator. You will be given a conversation between a user "
    "and one or two AI assistants, along with an evaluation rubric. "
    "Score the response(s) according to each criterion in the rubric. "
    "Return a JSON object where each key is a criterion name and the value is the "
    "score from the criterion's accepted_values list. "
    "Be objective, consistent, and precise. Return only the JSON object."
)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    match = re.search(r"\{[\s\S]+\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_quantifier(
    conversations: list[dict[str, Any]],
    criteria: dict[str, Any],
    *,
    model: str,
    api_key: str | None,
    num_seeds: int = 3,
    output_prefix: str | None = None,
) -> list[list[dict]]:
    if output_prefix:
        os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    criteria_str = json.dumps(criteria, indent=2)
    all_seed_results: list[list[dict]] = []

    for seed in range(num_seeds):
        seed_results = []

        for item in conversations:
            conv_id = item.get("id", "unknown") if isinstance(item, dict) else "unknown"
            conv_text = item.get("text", str(item)) if isinstance(item, dict) else str(item)

            prompt = (
                f"Evaluation rubric:\n{criteria_str}\n\n"
                f"Conversation to evaluate:\n{conv_text}\n\n"
                "Score the above conversation according to every criterion in the rubric."
            )

            raw = run_assistant_sync(
                name=f"quantifier_seed{seed}",
                system_message=QUANTIFIER_SYSTEM_MESSAGE,
                model=model,
                api_key=api_key,
                task=prompt,
                seed=seed,
            )
            scores = _extract_json(raw)

            seed_results.append({"id": conv_id, "scores": scores, "raw": raw})

        all_seed_results.append(seed_results)

        if output_prefix:
            out_path = f"{output_prefix}{seed}.json"
            with open(out_path, "w") as fh:
                json.dump(seed_results, fh, indent=2)
            print(f"[quantifier seed {seed}] {len(seed_results)} scored -> {out_path}")

    return all_seed_results
