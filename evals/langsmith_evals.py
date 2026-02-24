"""
LangSmith-based evaluation runner for Self-RAG pipeline.

Uploads the eval dataset to LangSmith, runs the graph as an experiment,
and scores each result with custom evaluators. Results are viewable
and comparable in the LangSmith UI.

Usage:
    python -m evals.langsmith_evals                              # run with default experiment name
    python -m evals.langsmith_evals --name "chunk600-topk4"      # name the experiment
    python -m evals.langsmith_evals --upload-only                # just upload the dataset
"""

import argparse
import json
from pathlib import Path

from langsmith import Client

from app.config import GRAPH_RECURSION_LIMIT
from app.graph import build_graph

# ── Paths ────────────────────────────────────────────────────────────────────
EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.json"

DATASET_NAME = "Self-RAG Eval Dataset"

ls_client = Client()


# ── 1. Upload dataset to LangSmith ──────────────────────────────────────────
def upload_dataset() -> str:
    """Create (or update) the eval dataset in LangSmith. Returns the dataset name."""
    with open(DATASET_PATH) as f:
        raw = json.load(f)

    # Check if dataset already exists
    try:
        existing = ls_client.read_dataset(dataset_name=DATASET_NAME)
        print(f"  Dataset '{DATASET_NAME}' already exists (id={existing.id}).")
        print("  Delete it in the UI or use a new name to re-upload.")
        return DATASET_NAME
    except Exception:
        pass  # dataset doesn't exist yet, create it

    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="20-question eval set for NovaMind AI Self-RAG pipeline.",
    )

    examples = []
    for q in raw:
        examples.append(
            {
                "inputs": {
                    "question": q["question"],
                },
                "outputs": {
                    "expected_answer_keywords": q.get("expected_answer_keywords", []),
                    "expected_need_retrieval": q.get("expected_need_retrieval"),
                    "expected_fallback": q.get("expected_fallback", False),
                    "category": q.get("category", ""),
                    "difficulty": q.get("difficulty", ""),
                    "source_docs": q.get("source_docs", []),
                },
            }
        )

    ls_client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"  ✅ Uploaded {len(examples)} examples to '{DATASET_NAME}'.")
    return DATASET_NAME


# ── 2. Target function (what LangSmith will call per example) ────────────────
def build_target():
    """Return a target function that runs the Self-RAG graph."""
    graph = build_graph()

    def target(inputs: dict) -> dict:
        """Run the Self-RAG pipeline on a single question."""
        initial_state = {
            "question": inputs["question"],
            "retries": 0,
            "rewrite_tries": 0,
        }
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        return {
            "answer": result.get("answer", ""),
            "need_retrieval": result.get("need_retrieval"),
            "is_supported": result.get("is_supported", ""),
            "is_use": result.get("is_use", ""),
            "use_reason": result.get("use_reason", ""),
            "evidence": result.get("evidence", []),
            "rewrite_tries": result.get("rewrite_tries", 0),
        }

    return target


# ── 3. Custom evaluators ────────────────────────────────────────────────────

def keyword_hit_rate(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score: what fraction of expected keywords appear in the answer."""
    expected = reference_outputs.get("expected_answer_keywords", [])
    answer = outputs.get("answer", "")

    if not expected:
        score = 1.0
    else:
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected if kw.lower() in answer_lower)
        score = hits / len(expected)

    return {
        "key": "keyword_hit_rate",
        "score": round(score, 2),
        "comment": f"{int(score * len(expected)) if expected else 0}/{len(expected)} keywords found",
    }


def retrieval_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score: did the graph make the right retrieval decision?"""
    expected = reference_outputs.get("expected_need_retrieval")
    actual = outputs.get("need_retrieval")

    if expected is None:
        score = 1.0
        comment = "No expected retrieval value to check"
    elif actual == expected:
        score = 1.0
        comment = f"Correct: retrieval={actual}"
    else:
        score = 0.0
        comment = f"Wrong: got {actual}, expected {expected}"

    return {"key": "retrieval_correct", "score": score, "comment": comment}


def fallback_detection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score: for negative tests, did the graph gracefully say 'I don't know'?"""
    expected_fallback = reference_outputs.get("expected_fallback", False)
    answer = outputs.get("answer", "")

    if not expected_fallback:
        return {
            "key": "fallback_detection",
            "score": None,
            "comment": "Not a negative test — skipped",
        }

    fallback_phrases = [
        "no relevant",
        "not found",
        "unable to find",
        "don't have",
        "do not have",
        "not mentioned",
        "no information",
        "couldn't find",
        "could not find",
        "no answer",
        "not available",
    ]
    triggered = any(phrase in answer.lower() for phrase in fallback_phrases)

    return {
        "key": "fallback_detection",
        "score": 1.0 if triggered else 0.0,
        "comment": "Fallback triggered" if triggered else "Expected fallback but got confident answer",
    }


def hallucination_check(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score: was the answer fully supported by retrieved evidence?"""
    is_supported = outputs.get("is_supported", "")
    need_retrieval = outputs.get("need_retrieval")

    if not need_retrieval:
        return {
            "key": "hallucination_check",
            "score": None,
            "comment": "No retrieval path — skipped",
        }

    # If the system correctly refused to answer, that's not a hallucination
    answer = outputs.get("answer", "")
    fallback_phrases = ["no relevant document", "no answer found", "unable to find"]
    if any(p in answer.lower() for p in fallback_phrases):
        return {
            "key": "hallucination_check",
            "score": 1.0,
            "comment": "Fallback answer — no hallucination",
        }

    score_map = {
        "fully_supported": 1.0,
        "partially_supported": 0.5,
        "not_supported": 0.0,
    }
    score = score_map.get(is_supported, 0.0)

    return {
        "key": "hallucination_check",
        "score": score,
        "comment": f"is_supported={is_supported}",
    }


def usefulness_check(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score: did the Self-RAG pipeline deem its own answer useful?"""
    is_use = outputs.get("is_use", "")
    need_retrieval = outputs.get("need_retrieval")

    if not need_retrieval:
        return {
            "key": "usefulness_check",
            "score": None,
            "comment": "No retrieval path — skipped",
        }

    score = 1.0 if is_use == "useful" else 0.0
    return {
        "key": "usefulness_check",
        "score": score,
        "comment": f"is_use={is_use}, reason={outputs.get('use_reason', '')}",
    }


ALL_EVALUATORS = [
    keyword_hit_rate,
    retrieval_correctness,
    fallback_detection,
    hallucination_check,
    usefulness_check,
]


# ── 4. Run experiment ───────────────────────────────────────────────────────
def run_experiment(experiment_name: str = "baseline"):
    """Upload dataset (if needed) and run a LangSmith experiment."""
    dataset_name = upload_dataset()

    print(f"\n  🚀 Running experiment: '{experiment_name}'")
    print(f"  Dataset: {dataset_name}")
    print(f"  Evaluators: {[e.__name__ for e in ALL_EVALUATORS]}\n")

    target = build_target()

    results = ls_client.evaluate(
        target,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )

    print("\n  ✅ Experiment complete!")
    print(f"  View results at: https://smith.langchain.com\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run Self-RAG evals via LangSmith"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Experiment name (e.g., 'chunk600-topk4', 'gpt4o-run')",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload the dataset, don't run the experiment",
    )
    args = parser.parse_args()

    if args.upload_only:
        upload_dataset()
    else:
        run_experiment(experiment_name=args.name)


if __name__ == "__main__":
    main()
