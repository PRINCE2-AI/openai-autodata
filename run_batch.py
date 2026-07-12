import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.agentic_loop import batch_process

load_dotenv()


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenAI Autodata Agentic Self-Instruct pipeline.")
    parser.add_argument("--papers-dir", default="data/raw_text")
    parser.add_argument("--max-papers", type=non_negative_int, default=10)
    parser.add_argument("--max-rounds", type=positive_int, default=8)
    parser.add_argument("--weak-rollouts", type=positive_int, default=2)
    parser.add_argument("--strong-rollouts", type=positive_int, default=2)
    parser.add_argument("--weak-max", type=float, default=0.55)
    parser.add_argument("--strong-min", type=float, default=0.65)
    parser.add_argument("--gap-min", type=float, default=0.15)
    parser.add_argument("--paper-delay", type=non_negative_float, default=1.0)
    parser.add_argument("--no-quality-check", action="store_true")
    parser.add_argument(
        "--budget-usd",
        type=positive_float,
        default=None,
        help="Total cost cap; defaults to OPENAI_BUDGET_USD or $50.",
    )
    parser.add_argument(
        "--budget-safety-margin-usd",
        type=non_negative_float,
        default=None,
        help="Unused budget buffer; defaults to OPENAI_BUDGET_SAFETY_USD or $2.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OPENAI AUTODATA BATCH PROCESSING")
    print("=" * 60)

    try:
        accepted, rejected = batch_process(
            papers_dir=args.papers_dir,
            max_papers=args.max_papers,
            max_rounds=args.max_rounds,
            weak_rollouts=args.weak_rollouts,
            strong_rollouts=args.strong_rollouts,
            weak_solver_avg_max=args.weak_max,
            strong_solver_avg_min=args.strong_min,
            gap_min=args.gap_min,
            quality_check=not args.no_quality_check,
            paper_delay=args.paper_delay,
            budget_usd=args.budget_usd,
            budget_safety_margin_usd=args.budget_safety_margin_usd,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except OpenAIError as exc:
        print(f"OpenAI API stopped the batch: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    total = len(accepted) + len(rejected)
    rate = (len(accepted) / total * 100) if total else 0.0
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Success Rate: {rate:.1f}%")
    if accepted:
        print(f"Accepted papers: {accepted}")
