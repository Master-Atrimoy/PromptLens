# 🔍 PromptLens

> **Semantic diff for LLM prompts** — understand what actually changed in intent, how different models respond, and which version performs better.
>
> Compare prompt versions beyond text diffs using embeddings, multi-model outputs, and an AI judge — all running locally via Ollama.

**Built for:** prompt engineers, LLM app developers, and anyone iterating on prompts.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-local-black)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green)
![Hydra](https://img.shields.io/badge/Hydra-config-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The problem

Most prompt iteration tools show you a character-level diff — essentially `git diff` for text. That tells you _what changed_, not whether the _intent changed_.
When working with LLMs, small wording tweaks can lead to large behavioral shifts. Without deeper analysis, you're guessing.

---

## The solution

PromptLens analyzes prompt changes at three levels:

- **Intent shift** — did the meaning change?
- **Model behavior** — how do different models respond?
- **Outcome quality** — which version is actually better?

---

## Capabilities

- **Three-layer semantic assessment** — prompt intent score, per-model output shift score, anatomy breakdown
- **Local model discovery** — auto-detects all pulled Ollama models; no manual config needed
- **Parallel inference** — models run concurrently via `ThreadPoolExecutor`; v1/v2 pairs run sequentially per model to avoid GPU contention
- **Anatomy tagger** — classifies every changed segment by prompt component type
- **LLM judge** — structured verdict (intent change / gained / lost / recommendation) from a local model
- **Graceful fallback** — if `nomic-embed-text` isn't installed, falls back to TF-IDF cosine similarity automatically
- **Dual interface** — Streamlit UI for interactive use, Typer CLI for scripting/CI pipelines
- **Pydantic v2 schemas** — all data models typed and validated end-to-end
- **Hydra config** — all thresholds, model defaults, and inference settings in `conf/config.yaml`

---

## Architecture

```
prompt-diff-visualizer/
├── core/
│   ├── schemas.py        # Pydantic v2 data models (source of truth)
│   ├── config.py         # Hydra config loader with typed defaults
│   ├── ollama_client.py  # Model discovery, inference, embeddings
│   ├── differ.py         # Structural diff + anatomy tagger (no models needed)
│   ├── embedder.py       # nomic-embed-text wrapper + TF-IDF fallback
│   ├── runner.py         # Async parallel inference across models
│   ├── judge.py          # LLM judge meta-prompt + JSON verdict parser
│   └── pipeline.py       # Orchestration — wires all modules together
├── app/
│   └── streamlit_app.py  # Full Streamlit UI
├── conf/
│   └── config.yaml       # Hydra config (thresholds, models, inference)
├── examples/
│   ├── summarization/    # v1.txt + v2.txt prompt pair
│   ├── classification/
│   └── roleplay/
├── tests/
│   ├── test_differ.py
│   ├── test_embedder.py
│   └── test_schemas.py
├── cli.py                # Typer CLI entrypoint
└── pyproject.toml
```

### Semantic pipeline (step by step)

```
Prompt v1 ──┐
            ├──► Structural diff + anatomy tagger ──► PromptDiff
Prompt v2 ──┘
            │
            ├──► Embed both prompts (nomic-embed-text / TF-IDF fallback)
            │    └──► Cosine similarity ──► PromptSemanticResult
            │
            ├──► Parallel inference (all selected models)
            │    ├── Model A: output_v1, output_v2
            │    ├── Model B: output_v1, output_v2
            │    └── Model C: output_v1, output_v2
            │         └──► Embed outputs ──► OutputSemanticResult per model
            │
            └──► LLM judge (Mistral/Llama3) ──► JudgeVerdict
```

---

## Shift level classification

| Score range | Level          | Meaning                                      |
| ----------- | -------------- | -------------------------------------------- |
| ≥ 0.95      | ✅ Trivial     | Wording only — meaning unchanged             |
| 0.85–0.95   | 🔵 Minor       | Light rewording — minor intent shift         |
| 0.65–0.85   | 🟡 Moderate    | Notable intent shift — review carefully      |
| 0.40–0.65   | 🟠 Major       | Significant restructure — different approach |
| < 0.40      | 🔴 Fundamental | Entirely different prompt                    |

Thresholds are configurable in `conf/config.yaml`.

---

## Prerequisites

**Python 3.11+**

**[Ollama](https://ollama.com)** installed and running locally

Pull at least one inference model and the embedding model:

```bash
ollama pull llama3
ollama pull mistral
ollama pull gemma
ollama pull nomic-embed-text   # recommended for best semantic scoring
```

> **Note:** If `nomic-embed-text` is not available, the tool automatically falls back to TF-IDF cosine similarity. All core features still work.

---

## Installation

```bash
git clone https://github.com/yourusername/prompt-diff-visualizer
cd prompt-diff-visualizer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

**What you'll see:**

1. **Sidebar** — live model list pulled from your local Ollama, model selector, judge toggle
2. **Input panel** — side-by-side prompt text areas (or upload `.txt` files), 3 built-in example pairs
3. **Semantic score** — colour-coded shift badge with similarity and shift values
4. **Structural diff** — annotated diff with anatomy tags inline
5. **Per-model comparison** — expandable output panels per model with shift scores
6. **Judge verdict** — intent change / gained / lost / recommendation

### CLI

```bash
# List all locally available models
python cli.py --list-models

# Diff two prompt files
python cli.py --v1 examples/summarization/v1.txt --v2 examples/summarization/v2.txt

# Specify models
python cli.py --v1 examples/roleplay/v1.txt --v2 examples/roleplay/v2.txt \
  --model llama3 --model mistral

# Output full JSON report
python cli.py --v1 examples/classification/v1.txt --v2 examples/classification/v2.txt \
  --json > report.json

# Skip the judge (faster)
python cli.py --v1 p1.txt --v2 p2.txt --no-judge
```

---

## Configuration

All settings live in `conf/config.yaml` (managed by Hydra):

```yaml
ollama:
  base_url: "http://localhost:11434" # Change for remote Ollama
  embedding_model: "nomic-embed-text"
  judge_model: "mistral" # Any local model for the judge

scoring:
  thresholds:
    trivial: 0.95
    minor: 0.85
    moderate: 0.65
    major: 0.40

inference:
  temperature: 0.7
  max_tokens: 1024
  parallel: true # false = sequential (debug mode)
```

Override any value at runtime via Hydra CLI syntax:

```bash
python cli.py --v1 p1.txt --v2 p2.txt ollama.judge_model=llama3
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover: anatomy classifier, structural differ, cosine similarity, TF-IDF fallback, and Pydantic schema validation. No Ollama connection required for tests.

---

## Example output

**Prompt pair:** `examples/roleplay/` — "Tell me about climate change." → detailed scientist persona prompt

```
🧠 Prompt Semantic Score
╔══════════════════════════════════════════════════════╗
║  FUNDAMENTAL SHIFT                                   ║
║  Entirely different prompt                           ║
║  Similarity: 0.312  |  Shift score: 0.688            ║
║  Embedding: nomic-embed-text                         ║
╚══════════════════════════════════════════════════════╝

📐 Anatomy Changes
┌────────────┬────────────────┬──────────────────────────────────────────┐
│ Type       │ Change         │ Text                                     │
├────────────┼────────────────┼──────────────────────────────────────────┤
│ role       │ added          │ You are a climate scientist presenting... │
│ constraint │ added          │ Use formal language. Avoid jargon.       │
│ instruction│ changed_to     │ Explain the three most critical...       │
└────────────┴────────────────┴──────────────────────────────────────────┘

⚖️  Judge verdict (mistral)
  Intent change: v2 narrows to expert scientific communication for a policy audience
  Gained:        Precision, authority, structured evidence requirement
  Lost:          Accessibility for general audiences
  Recommendation: context_dependent
```

---

## Extending the project

| What to add                      | Where                                            |
| -------------------------------- | ------------------------------------------------ |
| New anatomy pattern              | `core/differ.py` → `ANATOMY_PATTERNS`            |
| New shift threshold              | `conf/config.yaml` → `scoring.thresholds`        |
| New scoring metric (BLEU, ROUGE) | `core/scorer.py` (new module)                    |
| Save reports to disk             | `core/pipeline.py` + `cli.py --save` flag        |
| Multi-turn prompt diffs          | Extend `core/schemas.py` with `ConversationDiff` |

---

## Tech stack

| Layer            | Library                                       |
| ---------------- | --------------------------------------------- |
| Data validation  | Pydantic v2                                   |
| Configuration    | Hydra + OmegaConf                             |
| LLM inference    | Ollama (local) via httpx                      |
| Async execution  | asyncio + httpx AsyncClient                   |
| UI               | Streamlit                                     |
| CLI              | Typer + Rich                                  |
| Semantic scoring | nomic-embed-text (Ollama) + cosine similarity |
| Fallback scoring | TF-IDF (stdlib only)                          |
| Testing          | pytest                                        |

---

## License

MIT — use freely, attribution appreciated.
