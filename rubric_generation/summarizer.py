"""Checkpoint 2: Merge per-seed criteria into a single consolidated rubric."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from .autogen_compat import run_assistant_sync

SUMMARIZER_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You suggest criteria for evaluating different tasks. "
    "They should be distinguishable, quantifiable and not redundant. "
    'A criteria dictionary has keys = criteria names, values = {"description": ..., '
    '"accepted_values": ...}. '
    "Given a list of criteria dicts, pick the best 25 distinct criteria with the best "
    "descriptions and accepted_values. "
    "Return only the merged dictionary in JSON format."
)


def _extract_json(text: str) -> dict[str, Any] | None:
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


def _summarize_chunk(
    task: dict,
    crit_dicts: list,
    *,
    model: str,
    api_key: str | None,
) -> str:
    msg = (
        f"Task: {task['name']}.\n"
        f"Task description: {task['description']}\n"
        f"Suggested criteria: {crit_dicts}\n"
    )
    return run_assistant_sync(
        name="criteria_summarizer",
        system_message=SUMMARIZER_SYSTEM_MESSAGE,
        model=model,
        api_key=api_key,
        task=msg,
    )


def get_final_criteria(
    task: dict,
    crit_dicts: list[dict],
    *,
    model: str,
    api_key: str | None,
    chunk_size: int = 25,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Hierarchically merge criteria dicts into one rubric JSON."""
    intermediate: list = []
    if len(crit_dicts) > chunk_size:
        for i in range(0, len(crit_dicts), chunk_size):
            chunk = crit_dicts[i : i + chunk_size]
            intermediate.append(_summarize_chunk(task, chunk, model=model, api_key=api_key))
            print(f"[summarizer] chunk {i}-{i+len(chunk)-1} done")
        final_raw = _summarize_chunk(task, intermediate, model=model, api_key=api_key)
    else:
        final_raw = _summarize_chunk(task, crit_dicts, model=model, api_key=api_key)

    final_dict = _extract_json(final_raw)
    if final_dict is None:
        print("[summarizer] WARNING: could not parse final JSON.\n" + final_raw[:500],
              file=sys.stderr)
        final_dict = {}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(final_dict, fh, indent=2)
        print(f"[summarizer] saved {len(final_dict)} criteria -> {output_path}")

    return final_dict
