# AgentEval Replication — Preference / Chat Rubric Generation

Automated rubric generation pipeline for evaluating AI assistant responses,
applied to these datasets:

- [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k)
- [`Vezora/Code-Preference-Pairs`](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs)
- [`HumanLLMs/Human-Like-DPO-Dataset`](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)
- [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) (full `chosen` / `rejected` transcripts)
- [`argilla/ultrafeedback-binarized-preferences`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) (`instruction`, `chosen_response`, `rejected_response`; UltraFeedback pairs binarized from mean aspect ratings)

Adapted from the AutoGen-based criteria generation pipeline in
`naacl2025submission/scaling_and_verification/criteria_generation/`.

## Directory structure

```
AgentEval Replication/
├── run_rubric_generation.py   # CLI entry point
├── requirements.txt
├── .env.example               # copy to .env and add your API key
├── rubric_generation/
│   ├── config.py              # all tuneable parameters and paths
│   ├── data_loading.py        # HuggingFace dataset helpers
│   ├── critic.py              # Checkpoint 1: multi-seed critic agent
│   ├── summarizer.py          # Checkpoint 2: criteria merging agent
│   ├── quantifier.py          # Checkpoint 3: optional scoring agent
│   └── pipeline.py            # high-level orchestrator
└── experiment_outputs/
    ├── lmarena-ai__arena-expert-5k/
    ├── vezora__code-preference-pairs/
    ├── humanllms__human-like-dpo-dataset/
    ├── anthropic__hh-rlhf/
    └── argilla__ultrafeedback-binarized-preferences/
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

### Troubleshooting: AutoGen imports

Older versions (`pyautogen` 0.2.x) supported `import autogen`. **`pyautogen` 0.10+ installs the modern AgentChat stack** (`autogen_agentchat`), and **there is generally no usable top-level `autogen` import** from that package anymore.

Install exactly what `requirements.txt` pins (especially `pyautogen` + `autogen-ext[openai]`) using **the interpreter you run**:

```bash
python -m pip install -r requirements.txt
python -m pip show pyautogen autogen-ext autogen-agentchat
```

If you still want the legacy behavior, recreate a separate environment with **`pyautogen` 0.2.x**, but note it typically pins Python `<3.13`.

## Running the pipeline

### Full run (critic + summarizer)
```bash
python run_rubric_generation.py --api-key sk-...
```

### Run all supported datasets independently
```bash
python run_rubric_generation.py --api-key sk-... \
  --datasets "lmarena-ai/arena-expert-5k,Vezora/Code-Preference-Pairs,HumanLLMs/Human-Like-DPO-Dataset,Anthropic/hh-rlhf,argilla/ultrafeedback-binarized-preferences"
```

### Quick test (3 seeds, 100 dataset rows)
```bash
python run_rubric_generation.py --api-key sk-... \
    --num-critic-seeds 3 --max-samples 100
```

### Skip critic, re-run summarizer only
```bash
python run_rubric_generation.py --api-key sk-... --skip-critic
```

### Full run + quantifier on 50 battles
```bash
python run_rubric_generation.py --api-key sk-... --run-quantifier --quantifier-max-samples 50
```

### All options
```
--api-key               OpenAI API key (or set OPENAI_API_KEY env var)
--datasets              Comma-separated dataset names to run independently
--dataset-split         Dataset split to load (default: train)
--reset-outputs         Delete each dataset's `experiment_outputs/<slug>/` tree before running (clean regen)
--model                 LLM model name (default: gpt-4o-mini)
--num-critic-seeds      Independent critic runs (default: 15)
--num-quantifier-seeds  Quantifier scoring passes (default: 3)
--max-samples           Cap dataset rows for faster testing
--skip-critic           Load existing seed files instead of re-running
--skip-summarizer       Load existing final_criteria.json
--run-quantifier        Also score conversations with the rubric
--quantifier-max-samples Max battles to score (default: 50)
```

## How the pipeline works

1. **Data loading** — Pulls whichever dataset(s) you pass to `--datasets`
   (`lmarena-ai/arena-expert-5k`, `Vezora/Code-Preference-Pairs`, `HumanLLMs/Human-Like-DPO-Dataset`, `Anthropic/hh-rlhf`, or `argilla/ultrafeedback-binarized-preferences`)
   from Hugging Face and formats rows into prompts for criterion generation.

   For `lmarena-ai/arena-expert-5k`, each row is a head-to-head battle between two LLMs judged by an expert.
   For preference-pair datasets, rows are modeled as preference comparisons over prompts/responses.
   For `Anthropic/hh-rlhf`, each row has full multi-turn transcripts in `chosen` vs `rejected` (helpfulness / harmlessness preference data).
   For `argilla/ultrafeedback-binarized-preferences`, each row compares `chosen_response` vs `rejected_response` for the same `instruction`.

2. **Critic (Checkpoint 1)** — Runs an AutoGen `AssistantAgent` N times with
   different `cache_seed` values. Each run proposes a distinct set of evaluation
   criteria in JSON format. Results saved to dataset-specific `experiment_outputs/<dataset>/criteria/`.

3. **Summarizer (Checkpoint 2)** — A second agent merges all per-seed criteria
   dicts into a single rubric with ≤25 distinct, well-described criteria.
   Result saved to dataset-specific `experiment_outputs/<dataset>/final_criteria.json`.

4. **Quantifier (Checkpoint 3, optional)** — Scores a sample of rows against
   the final rubric. Results saved to dataset-specific `experiment_outputs/<dataset>/evaluated_problems-{seed}.json`.

## Dataset fields used

| Field | Used for |
|---|---|
| `conversation_a / b` | The two model responses being evaluated |
| `model_a / b` | Model names (for labelling) |
| `winner` | Identifies successful vs. unsuccessful examples |
| `language` | Available for filtering (default: all languages) |
| `occupational_tags` | Available for domain-specific analysis |
| `chosen` / `rejected` | Preferred vs alternate full transcripts (`Anthropic/hh-rlhf` and prompt-based DPO rows) |
| `instruction` / `chosen_response` / `rejected_response` | Same-instruction preference pairs (`argilla/ultrafeedback-binarized-preferences`) |
