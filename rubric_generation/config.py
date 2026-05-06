"""
Configuration for the multi-dataset rubric generation pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ── Dataset ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET_NAME = "lmarena-ai/arena-expert-5k"
DATASET_SPLIT = "train"
SUPPORTED_DATASETS = (
    "lmarena-ai/arena-expert-5k",
    "Vezora/Code-Preference-Pairs",
    "HumanLLMs/Human-Like-DPO-Dataset",
    "Anthropic/hh-rlhf",
    "argilla/ultrafeedback-binarized-preferences",
)

# ── Pipeline hyper-params ──────────────────────────────────────────────────────
NUM_CRITIC_SEEDS = 15       # how many independent critic runs to sample criteria
NUM_QUANTIFIER_SEEDS = 3    # seeds for the quantifier stage (evaluation)
SUMMARIZER_CHUNK_SIZE = 25  # max criteria dicts per summarizer call

# ── Task definitions (per dataset; used by critic + summarizer) ───────────────
#
# IMPORTANT: Previously the pipeline reused a generic chat-evaluation prompt for every
# dataset, which pushes the merged rubrics to look alike. Keep these sharply distinct.


def _arena_task() -> tuple[str, str]:
    name = "Arena-style evaluation of assistant responses in multi-turn chat"
    description = """\
You are an expert judge evaluating AI assistant outputs in realistic, multi-turn \
chats with side-by-side model comparisons (as in LM Arena-style preference data).

Your evaluation should prioritize what makes one assistant answer better than another \
in practice: correctness, relevance, helpfulness, clarity, completeness, safety, \
instruction-following, and reasoning quality.

When two candidates are shown, judge which better serves the user's goals and explain \
the tradeoffs (not just surface fluency).\
"""
    return name, description


def _code_preference_pairs_task() -> tuple[str, str]:
    name = "Preference judgment for code-assistant outputs (bug sensitivity + engineering quality)"
    description = """\
You are evaluating pairs of candidate solutions for coding tasks, as in the \
Code-Preference-Pairs setting used for Direct Preference Optimization (DPO): each row \
contrasts a preferred answer vs a rejected answer for the same instruction (+ optional \
I/O context), where differences often correspond to subtle bugs or quality regressions \
introduced by a synthetic edit process (see the dataset card for the Open-Critic-GPT / \
Code-Preference-Pairs construction story).

Focus on criteria that matter for real software engineering: functional correctness on \
the stated requirements, edge cases, efficiency (when relevant), robustness, API/library \
usage, readability/maintainability, and safe handling of errors/security footguns. \
When code is present, prefer concrete, testable signals over generic “helpfulness”.

Dataset context: https://huggingface.co/datasets/Vezora/Code-Preference-Pairs \
(related discussion of the broader pipeline appears alongside the Open-Critic-GPT line).\
"""
    return name, description


def _human_like_dpo_task() -> tuple[str, str]:
    name = "Human-like conversational quality vs formal assistant tone (DPO preference modeling)"
    description = """\
You are judging responses where one candidate is deliberately more conversational and \
humane, and the other resembles a cautious, formal assistant voice—matching the Human-Like \
DPO setting described in the dataset card and the associated write-up “Enhancing Human-Like \
Responses in Large Language Models” (arXiv:2501.05032; see also the Hugging Face paper page).

Emphasize natural dialogue: warmth, engagement, appropriate personality/emoji use (when \
suitable), coherence, listening/turn-taking, and avoiding unnecessary boilerplate \
disclaimers or stiff corporate phrasing—while still being accurate, safe, and helpful.

Do not optimize for “most formal”; optimize for the dataset’s axis: human-like engagement \
with responsible assistance.

Dataset context: https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset \
Paper pointer: https://arxiv.org/abs/2501.05032 \
"""
    return name, description


def _ultrafeedback_binarized_task() -> tuple[str, str]:
    name = "UltraFeedback-style instruction-following preference over diverse completions (mean-rated binarization)"
    description = """\
You are judging two candidate assistant answers to the same user instruction, drawn from UltraFeedback-derived \
preference data binarized by Argilla: `chosen_response` vs `rejected_response` pairs where the preference side \
was picked from multi-aspect human ratings averaged across honesty, instruction-following, truthfulness-style \
axes (not a single synthetic overall critique score).

Reward answers that better satisfy the instruction: factual accuracy when claims matter, appropriate scope \
and completeness, coherent reasoning, safe behavior, and faithful adherence to formatting or constraints. Both \
sides may look fluent; look for substantive gaps, subtle instruction misses, hedging or overreach, and clarity.

Dataset context: https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences \
"""
    return name, description


def _hh_rlhf_task() -> tuple[str, str]:
    name = "Human preference judgments for assistant dialogues (RLHF-style transcripts)"
    description = """\
You are judging full multi-turn transcripts where human labelers preferred one assistant \
completion over another—the Anthropic HH-RLHF setting used to train harmless and helpful \
assistants via RL from human feedback (see Training a Helpful and Harmless Assistant with \
RLHF, arXiv:2204.05862).

Each row pairs a preferred vs rejected completion for the same user context (both sides appear \
as full \"Human: ... Assistant: ...\" style transcripts). Infer what makes one trajectory \
better aligned with helpful, honest, and harmless assistance: staying on task, admitting \
limits, refusing unsafe requests appropriately, factual care, clarity, tone, and not over-claiming.

Dataset context: https://huggingface.co/datasets/Anthropic/hh-rlhf \
"""
    return name, description


_DATASET_TASKS: dict[str, tuple[str, str]] = {
    "lmarena-ai/arena-expert-5k": _arena_task(),
    "Vezora/Code-Preference-Pairs": _code_preference_pairs_task(),
    "HumanLLMs/Human-Like-DPO-Dataset": _human_like_dpo_task(),
    "Anthropic/hh-rlhf": _hh_rlhf_task(),
    "argilla/ultrafeedback-binarized-preferences": _ultrafeedback_binarized_task(),
}


def get_task_definition(dataset_name: str) -> tuple[str, str]:
    if dataset_name not in _DATASET_TASKS:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            f"Expected one of {list(_DATASET_TASKS)}"
        )
    return _DATASET_TASKS[dataset_name]


# Back-compat for older imports: arena-expert task as default.
TASK_NAME, TASK_DESCRIPTION = _arena_task()

# ── Output paths ───────────────────────────────────────────────────────────────
# Default to a local folder in this repository.
# Override by setting OUTPUT_DIR env var.
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiment_outputs")
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

def dataset_to_slug(dataset_name: str) -> str:
    """Convert HF dataset names into safe folder names."""
    return dataset_name.replace("/", "__").replace(" ", "_").lower()


def get_dataset_output_paths(dataset_name: str) -> dict[str, str]:
    """
    Build dataset-specific output paths under OUTPUT_DIR.

    Example:
      experiment_outputs/lmarena-ai__arena-expert-5k/...
    """
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_to_slug(dataset_name))
    return {
        "dataset_dir": dataset_dir,
        "critic_output_prefix": os.path.join(dataset_dir, "criteria", "arena-rubric-"),
        "final_criteria_path": os.path.join(dataset_dir, "final_criteria.json"),
        "quantifier_output_prefix": os.path.join(dataset_dir, "evaluated_problems-"),
    }
