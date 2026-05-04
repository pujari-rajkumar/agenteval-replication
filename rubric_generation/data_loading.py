"""Data loading and row-formatting utilities for supported datasets."""
from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset


def load_arena_dataset(
    dataset_name: str = "lmarena-ai/arena-expert-5k",
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return list(ds)


def _format_dpo_row(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
    user_input = str(row.get("input", "")).strip()
    chosen = str(row.get("chosen", row.get("accepted", ""))).strip()
    rejected = str(row.get("rejected", "")).strip()

    user_parts = [p for p in [prompt, user_input] if p]
    user_text = "\n".join(user_parts).strip()

    # HH-RLHF-style rows: only full transcript strings (no isolated prompt field).
    if not user_text and chosen and rejected:
        return (
            f"=== Preferred transcript ===\n{chosen}\n\n"
            f"=== Rejected transcript ===\n{rejected}"
        )

    return (
        f"=== User Prompt ===\n{user_text}\n\n"
        f"=== Preferred Response ===\n{chosen}\n\n"
        f"=== Rejected Response ===\n{rejected}"
    )


def parse_conversation(raw_conv: Any) -> list[dict[str, str]]:
    """Parse a conversation field into list of {role, content} dicts."""
    if isinstance(raw_conv, str):
        try:
            raw_conv = json.loads(raw_conv)
        except json.JSONDecodeError:
            return [{"role": "user", "content": raw_conv}]

    messages = []
    for turn in raw_conv:
        role = turn.get("role", "user") if isinstance(turn, dict) else str(turn[0])
        content_field = (
            turn.get("content", "") if isinstance(turn, dict) else turn[1]
        )
        if isinstance(content_field, (list, tuple)):
            parts = []
            for p in content_field:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = " ".join(parts).strip()
        else:
            content = str(content_field)
        messages.append({"role": role, "content": content})
    return messages


def format_conversation_for_eval(row: dict[str, Any], which: str = "a") -> str:
    messages = parse_conversation(row[f"conversation_{which}"])
    model_name = row.get(f"model_{which}", f"Model {which.upper()}")
    lines = [f"[Model: {model_name}]"]
    for msg in messages:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines)


def format_battle_for_eval(row: dict[str, Any]) -> str:
    # Arena dataset format.
    if "conversation_a" in row and "conversation_b" in row:
        conv_a = format_conversation_for_eval(row, "a")
        conv_b = format_conversation_for_eval(row, "b")
        winner = row.get("winner", "unknown")
        return (
            f"=== Model A ===\n{conv_a}\n\n"
            f"=== Model B ===\n{conv_b}\n\n"
            f"=== Expert Judgment: {winner} ==="
        )

    # Preference-pair datasets with chosen/rejected (or accepted/rejected).
    if ("chosen" in row or "accepted" in row) and "rejected" in row:
        return _format_dpo_row(row)

    # Fallback: keep pipeline robust if row schema drifts.
    return json.dumps(row, ensure_ascii=True, default=str)


def sample_examples(
    rows: list[dict[str, Any]],
    n_success: int = 2,
    n_failure: int = 2,
) -> dict[str, list[str]]:
    """Return successful and unsuccessful examples across supported schemas."""
    successes, failures = [], []
    for row in rows:
        # Arena style rows with explicit winner labels.
        if "winner" in row and "conversation_a" in row and "conversation_b" in row:
            winner = row.get("winner", "")
            fmt = format_battle_for_eval(row)
            if winner in ("model_a", "model_b") and len(successes) < n_success:
                successes.append(fmt)
            elif winner in ("tie", "both_bad") and len(failures) < n_failure:
                failures.append(fmt)
        # Preference-pair rows: chosen/accepted is success, rejected is failure.
        elif ("chosen" in row or "accepted" in row) and "rejected" in row:
            prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
            user_input = str(row.get("input", "")).strip()
            user_parts = [p for p in [prompt, user_input] if p]
            user_text = "\n".join(user_parts).strip()
            chosen = str(row.get("chosen", row.get("accepted", ""))).strip()
            rejected = str(row.get("rejected", "")).strip()
            if not user_text and chosen and rejected:
                good, bad = (
                    f"=== Preferred transcript ===\n{chosen}",
                    f"=== Rejected transcript ===\n{rejected}",
                )
            else:
                good = (
                    f"=== User Prompt ===\n{user_text}\n\n=== Response ===\n{chosen}"
                )
                bad = (
                    f"=== User Prompt ===\n{user_text}\n\n=== Response ===\n{rejected}"
                )
            if len(successes) < n_success:
                successes.append(good)
            if len(failures) < n_failure:
                failures.append(bad)
        else:
            # Unknown row schema; ignore for seed examples.
            pass
        if len(successes) >= n_success and len(failures) >= n_failure:
            break
    return {"success": successes, "failure": failures}
