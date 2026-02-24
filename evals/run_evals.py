"""
Evaluation runner for Self-RAG pipeline.

Runs every question in evals/dataset.json through the compiled graph,
compares results against expectations, and writes a detailed report.

Usage:
    python -m evals.run_evals                # run all questions
    python -m evals.run_evals --ids 1 4 7    # run specific question ids
    python -m evals.run_evals --category pricing  # run by category
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import GRAPH_RECURSION_LIMIT
from app.graph import build_graph

# ── Paths ────────────────────────────────────────────────────────────────────
EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.json"
RESULTS_DIR = EVALS_DIR / "results"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_dataset(
    ids: list[int] | None = None,
    category: str | None = None,
) -> list[dict]:
    """Load and optionally filter the eval dataset."""
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if ids:
        dataset = [q for q in dataset if q["id"] in ids]
    if category:
        dataset = [q for q in dataset if q["category"] == category]

    return dataset


def keyword_hit_rate(expected_keywords: list[str], answer: str) -> float:
    """Return fraction of expected keywords found (case-insensitive) in answer."""
    if not expected_keywords:
        return 1.0  # nothing to check
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def evaluate_single(
    question_data: dict,
    graph,
) -> dict:
    """Run a single eval question and return a result dict."""
    qid = question_data["id"]
    question = question_data["question"]
    expected_retrieval = question_data.get("expected_need_retrieval")
    expected_keywords = question_data.get("expected_answer_keywords", [])
    expected_fallback = question_data.get("expected_fallback", False)

    print(f"  [{qid:>2}] {question}")

    initial_state = {
        "question": question,
        "retries": 0,
        "rewrite_tries": 0,
    }

    t0 = time.perf_counter()
    try:
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        elapsed = time.perf_counter() - t0
        error = None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        result = {}
        error = str(e)

    answer = result.get("answer", "")
    need_retrieval = result.get("need_retrieval")
    is_supported = result.get("is_supported", "")
    is_use = result.get("is_use", "")
    use_reason = result.get("use_reason", "")
    retrieval_query = result.get("retrieval_query", "")
    rewrite_tries = result.get("rewrite_tries", 0)
    evidence = result.get("evidence", [])

    # ── Checks ───────────────────────────────────────────────────────────
    retrieval_correct = (
        need_retrieval == expected_retrieval
        if expected_retrieval is not None
        else None
    )

    kw_rate = keyword_hit_rate(expected_keywords, answer)

    # For negative tests, the answer should acknowledge lack of info
    fallback_triggered = False
    if expected_fallback:
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
        fallback_triggered = any(
            phrase in answer.lower() for phrase in fallback_phrases
        )

    # Overall pass/fail
    passed = True
    fail_reasons: list[str] = []

    if error:
        passed = False
        fail_reasons.append(f"Runtime error: {error}")

    if retrieval_correct is False:
        passed = False
        fail_reasons.append(
            f"Retrieval decision wrong: got {need_retrieval}, expected {expected_retrieval}"
        )

    if expected_keywords and kw_rate < 0.5:
        passed = False
        fail_reasons.append(
            f"Keyword hit rate too low: {kw_rate:.0%} ({int(kw_rate * len(expected_keywords))}/{len(expected_keywords)})"
        )

    if expected_fallback and not fallback_triggered:
        passed = False
        fail_reasons.append("Expected fallback/no-answer but got a confident response")

    return {
        "id": qid,
        "question": question,
        "category": question_data.get("category", ""),
        "difficulty": question_data.get("difficulty", ""),
        "passed": passed,
        "fail_reasons": fail_reasons,
        # Graph outputs
        "answer": answer,
        "need_retrieval": need_retrieval,
        "is_supported": is_supported,
        "is_use": is_use,
        "use_reason": use_reason,
        "evidence": evidence,
        "retrieval_query": retrieval_query,
        "rewrite_tries": rewrite_tries,
        # Metrics
        "keyword_hit_rate": round(kw_rate, 2),
        "retrieval_correct": retrieval_correct,
        "fallback_triggered": fallback_triggered if expected_fallback else None,
        "latency_s": round(elapsed, 2),
        "error": error,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def run_evals(
    ids: list[int] | None = None,
    category: str | None = None,
) -> dict:
    """Run the full evaluation suite and return the summary + individual results."""
    dataset = load_dataset(ids=ids, category=category)
    if not dataset:
        print("No questions matched the given filters.")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Self-RAG Evaluation  |  {len(dataset)} questions")
    print(f"{'=' * 60}\n")

    graph = build_graph()
    results: list[dict] = []

    for q in dataset:
        r = evaluate_single(q, graph)
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"       → {status}  ({r['latency_s']}s)")
        if r["fail_reasons"]:
            for reason in r["fail_reasons"]:
                print(f"         ⚠ {reason}")
        print()
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    avg_latency = sum(r["latency_s"] for r in results) / total if total else 0
    avg_kw_rate = (
        sum(r["keyword_hit_rate"] for r in results) / total if total else 0
    )

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"] or "unknown"
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed / total:.0%}" if total else "N/A",
        "avg_latency_s": round(avg_latency, 2),
        "avg_keyword_hit_rate": round(avg_kw_rate, 2),
        "by_category": {
            cat: f"{v['passed']}/{v['total']}" for cat, v in categories.items()
        },
    }

    report = {"summary": summary, "results": results}

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed ({summary['pass_rate']})")
    print(f"  Avg latency: {avg_latency:.2f}s  |  Avg keyword hit: {avg_kw_rate:.0%}")
    print(f"{'─' * 60}")
    for cat, counts in categories.items():
        print(f"  {cat:25s}  {counts['passed']}/{counts['total']}")
    print(f"{'=' * 60}\n")

    # ── Save to disk ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved → {out_path}\n")

    return report


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run Self-RAG evaluations")
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="Run only these question IDs",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Run only questions in this category",
    )
    args = parser.parse_args()
    run_evals(ids=args.ids, category=args.category)


if __name__ == "__main__":
    main()
