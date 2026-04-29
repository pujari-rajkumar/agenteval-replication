"""Checkpoint 1: Run critic agent multiple times to generate diverse criteria."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from .autogen_compat import run_assistant_sync

CRITIC_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You suggest criteria for evaluating different tasks. "
    "They should be distinguishable, quantifiable and not redundant. "
    "Convert the evaluation criteria into a dictionary where the keys are the criteria. "
    'The value of each key is a dictionary as follows {"description": criteria description, '
    '"accepted_values": possible accepted inputs for this key}. '
    "Make sure the keys are criteria for assessing the given task. "
    '"accepted_values" include the acceptable inputs for each key that are fine-grained '
    "and preferably multi-graded levels. "
    '"description" includes the criterion description. '
    "Return only the dictionary in JSON format."
)


def build_sys_msg(task: dict[str, str]) -> str:
    parts = [f"Task: {task['name']}.", f"Task description: {task['description']}"]
    examples = task.get("examples", "").strip()
    if examples:
        parts.append(examples)
    return "\n".join(parts)


def extract_json_dict(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]+\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_critic_seeds(
    sys_msg: str,
    *,
    model: str,
    api_key: str | None,
    num_seeds: int,
    output_prefix: str,
) -> list[dict[str, Any]]:
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    crit_dicts: list[dict[str, Any]] = []

    for seed in range(num_seeds):
        raw = run_assistant_sync(
            name="critic",
            system_message=CRITIC_SYSTEM_MESSAGE,
            model=model,
            api_key=api_key,
            task=sys_msg,
            temperature=0.8,
            seed=seed,
        )

        out_path = f"{output_prefix}{seed}.json"
        crit_dict = extract_json_dict(raw)
        if crit_dict:
            with open(out_path, "w") as fh:
                json.dump(crit_dict, fh, indent=2)
            crit_dicts.append(crit_dict)
            print(f"[critic seed {seed}] saved {len(crit_dict)} criteria -> {out_path}")
        else:
            print(f"[critic seed {seed}] WARNING: could not parse JSON.\n{raw[:200]}",
                  file=sys.stderr)

    return crit_dicts


def load_critic_outputs(output_prefix: str, num_seeds: int) -> list[dict[str, Any]]:
    crit_dicts = []
    for seed in range(num_seeds):
        path = f"{output_prefix}{seed}.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            d = json.loads(raw)
            if d:
                crit_dicts.append(d)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[load_critic_outputs] skipping {path}: {exc}", file=sys.stderr)
    return crit_dicts
