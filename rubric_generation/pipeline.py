"""High-level pipeline for rubric generation across supported datasets."""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from .config import (
    DATASET_SPLIT,
    DEFAULT_DATASET_NAME,
    LLM_MODEL,
    NUM_CRITIC_SEEDS,
    NUM_QUANTIFIER_SEEDS,
    OPENAI_API_KEY,
    SUMMARIZER_CHUNK_SIZE,
    get_dataset_output_paths,
    get_task_definition,
    OUTPUT_DIR,
)
from .critic import build_sys_msg, load_critic_outputs, run_critic_seeds
from .data_loading import format_battle_for_eval, load_arena_dataset, sample_examples
from .summarizer import get_final_criteria


def build_task(dataset_name: str, examples: dict[str, list[str]]) -> dict[str, str]:
    task_name, task_description = get_task_definition(dataset_name)
    success_block = "\n\n".join(
        f"Successful Example #{i+1}:\n{ex}"
        for i, ex in enumerate(examples.get("success", []))
    )
    failure_block = "\n\n".join(
        f"Unsuccessful Example #{i+1}:\n{ex}"
        for i, ex in enumerate(examples.get("failure", []))
    )
    examples_text = "\n\n".join(filter(None, [success_block, failure_block]))
    return {"name": task_name, "description": task_description, "examples": examples_text}


def resolve_llm_settings(api_key: str | None = None, model: str | None = None) -> tuple[str, str]:
    resolved_model = model or LLM_MODEL
    resolved_key = api_key or OPENAI_API_KEY
    return resolved_model, resolved_key


def run_pipeline(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DATASET_SPLIT,
    api_key: str | None = None,
    model: str | None = None,
    num_critic_seeds: int = NUM_CRITIC_SEEDS,
    num_quantifier_seeds: int = NUM_QUANTIFIER_SEEDS,
    max_samples: int | None = None,
    skip_critic: bool = False,
    skip_summarizer: bool = False,
    run_quantifier: bool = False,
    quantifier_max_samples: int = 50,
    reset_outputs: bool = False,
) -> dict[str, Any]:
    """Run critic → summarizer → optional quantifier."""
    llm_model, llm_api_key = resolve_llm_settings(api_key, model)
    output_paths = get_dataset_output_paths(dataset_name)
    critic_output_prefix = output_paths["critic_output_prefix"]
    final_criteria_path = output_paths["final_criteria_path"]
    quantifier_output_prefix = output_paths["quantifier_output_prefix"]

    if reset_outputs:
        dataset_dir = output_paths["dataset_dir"]
        root = os.path.abspath(OUTPUT_DIR)
        ds_abs = os.path.abspath(dataset_dir)
        if ds_abs != root and not ds_abs.startswith(root + os.sep):
            raise ValueError(f"Refusing to reset_outputs outside OUTPUT_DIR: {dataset_dir}")
        if os.path.exists(dataset_dir):
            print(f"[pipeline] Removing existing outputs -> {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            print(f"[pipeline] reset_outputs requested but nothing to delete: {dataset_dir}")

    print(f"[pipeline] Loading {dataset_name} ...")
    rows = load_arena_dataset(dataset_name, dataset_split, max_samples)
    print(f"[pipeline] {len(rows)} rows loaded")

    examples = sample_examples(rows, n_success=2, n_failure=2)
    task = build_task(dataset_name, examples)
    sys_msg = build_sys_msg(task)

    if not skip_critic:
        print(f"[pipeline] Running {num_critic_seeds} critic seeds ...")
        crit_dicts = run_critic_seeds(
            sys_msg,
            model=llm_model,
            api_key=llm_api_key,
            num_seeds=num_critic_seeds,
            output_prefix=critic_output_prefix,
        )
    else:
        print("[pipeline] Loading existing critic outputs ...")
        crit_dicts = load_critic_outputs(critic_output_prefix, num_critic_seeds)
    print(f"[pipeline] {len(crit_dicts)} valid criteria dicts")

    if not skip_summarizer:
        criteria = get_final_criteria(
            task,
            crit_dicts,
            model=llm_model,
            api_key=llm_api_key,
            chunk_size=SUMMARIZER_CHUNK_SIZE,
            output_path=final_criteria_path,
        )
    else:
        if os.path.exists(final_criteria_path):
            with open(final_criteria_path) as fh:
                criteria = json.load(fh)
            print(f"[pipeline] Loaded criteria from {final_criteria_path}")
        else:
            criteria = {}
            print("[pipeline] WARNING: no final criteria file and skip_summarizer=True")

    quantifier_results = []
    if run_quantifier and criteria:
        from .quantifier import run_quantifier as _rq
        eval_rows = rows[:quantifier_max_samples]
        convs = [{"id": r.get("id", str(i)), "text": format_battle_for_eval(r)}
                 for i, r in enumerate(eval_rows)]
        print(f"[pipeline] Quantifying {len(convs)} battles ...")
        quantifier_results = _rq(
            convs,
            criteria,
            model=llm_model,
            api_key=llm_api_key,
            num_seeds=num_quantifier_seeds,
            output_prefix=quantifier_output_prefix,
        )

    return {
        "dataset_name": dataset_name,
        "task": task,
        "criteria": criteria,
        "quantifier_results": quantifier_results,
        "output_paths": output_paths,
    }
