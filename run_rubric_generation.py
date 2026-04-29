#!/usr/bin/env python3
"""
Entry point for the AgentEval replication rubric generation pipeline (multi-dataset).

Usage examples:

  # Full run (critic + summarizer) with default settings
  python run_rubric_generation.py --api-key sk-...

  # Critic only (skip summarizer)
  python run_rubric_generation.py --api-key sk-... --skip-summarizer

  # Skip critic (use existing seed files) and only re-run summarizer
  python run_rubric_generation.py --api-key sk-... --skip-critic

  # Full run + quantifier on first 50 battles
  python run_rubric_generation.py --api-key sk-... --run-quantifier

  # Faster test: only 3 critic seeds, 100 dataset rows
  python run_rubric_generation.py --api-key sk-... --num-critic-seeds 3 --max-samples 100

Environment:
  You can also set OPENAI_API_KEY and LLM_MODEL in a .env file or as shell env vars
  instead of passing --api-key.
"""
import argparse
import os
import sys

# Allow running from the repo root without installing as a package
sys.path.insert(0, os.path.dirname(__file__))

from rubric_generation.pipeline import run_pipeline
from rubric_generation.config import DEFAULT_DATASET_NAME, SUPPORTED_DATASETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate evaluation rubrics for supported Hugging Face preference/chat datasets."
    )
    p.add_argument(
        "--datasets",
        default=DEFAULT_DATASET_NAME,
        help=(
            "Comma-separated Hugging Face dataset names to run independently. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        ),
    )
    p.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var / .env file).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: gpt-4o-mini, or LLM_MODEL env var).",
    )
    p.add_argument(
        "--num-critic-seeds",
        type=int,
        default=None,
        help="Number of independent critic runs (default: 15).",
    )
    p.add_argument(
        "--num-quantifier-seeds",
        type=int,
        default=None,
        help="Number of quantifier scoring passes (default: 3).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap dataset rows for faster testing (default: use all).",
    )
    p.add_argument(
        "--skip-critic",
        action="store_true",
        help="Skip the critic stage and load existing seed files.",
    )
    p.add_argument(
        "--skip-summarizer",
        action="store_true",
        help="Skip the summarizer stage and load existing final_criteria.json.",
    )
    p.add_argument(
        "--run-quantifier",
        action="store_true",
        help="Run the quantifier stage after generating/loading criteria.",
    )
    p.add_argument(
        "--quantifier-max-samples",
        type=int,
        default=50,
        help="Max battles to score in the quantifier stage (default: 50).",
    )
    p.add_argument(
        "--reset-outputs",
        action="store_true",
        help=(
            "Before each dataset run, delete that dataset's folder under experiment_outputs/ "
            "(criteria seeds, final_criteria.json, quantifier JSON). Use for a clean regeneration."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kwargs: dict = {}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.model:
        kwargs["model"] = args.model
    if args.num_critic_seeds is not None:
        kwargs["num_critic_seeds"] = args.num_critic_seeds
    if args.num_quantifier_seeds is not None:
        kwargs["num_quantifier_seeds"] = args.num_quantifier_seeds
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    kwargs["skip_critic"] = args.skip_critic
    kwargs["skip_summarizer"] = args.skip_summarizer
    kwargs["run_quantifier"] = args.run_quantifier
    kwargs["quantifier_max_samples"] = args.quantifier_max_samples
    kwargs["dataset_split"] = args.dataset_split
    kwargs["reset_outputs"] = args.reset_outputs

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    invalid = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(
            "Unsupported dataset(s): "
            + ", ".join(invalid)
            + f". Supported: {', '.join(SUPPORTED_DATASETS)}"
        )

    for dataset_name in datasets:
        result = run_pipeline(dataset_name=dataset_name, **kwargs)
        criteria = result["criteria"]
        output_paths = result["output_paths"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Pipeline complete. Final rubric has {len(criteria)} criteria.")
        print(f"Outputs: {output_paths['dataset_dir']}")
        if criteria:
            print("Criteria names:", list(criteria.keys()))
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
