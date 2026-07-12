import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAIError
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.openai_agent import ChallengerAgent, JudgeAgent, StrongSolverAgent, WeakSolverAgent
from src.utils.cost_tracker import BudgetExceededError, configure_cost_tracker, get_cost_tracker


class InvalidQAError(ValueError):
    pass


class RolloutFailedError(RuntimeError):
    pass


def validate_qa_item(qa: Dict) -> None:
    if not isinstance(qa, dict):
        raise InvalidQAError("Challenger output must be a JSON object.")

    for field in ("question_type", "context", "question", "reference_answer"):
        if not isinstance(qa.get(field), str) or not qa[field].strip():
            raise InvalidQAError(f"Missing or empty '{field}'.")

    skills = qa.get("reasoning_skills")
    if not isinstance(skills, list) or len(skills) < 2 or not all(isinstance(skill, str) and skill.strip() for skill in skills):
        raise InvalidQAError("reasoning_skills must contain at least two non-empty strings.")

    rubric = qa.get("rubric")
    if not isinstance(rubric, list) or not 10 <= len(rubric) <= 15:
        count = len(rubric) if isinstance(rubric, list) else 0
        raise InvalidQAError(f"Rubric had {count} criteria; expected 10-15.")

    positive_count = 0
    negative_count = 0
    for index, item in enumerate(rubric, start=1):
        if not isinstance(item, dict):
            raise InvalidQAError(f"Rubric criterion {index} must be an object.")
        criterion = item.get("criterion")
        weight = item.get("weight")
        category = str(item.get("category", "")).strip().lower()
        if not isinstance(criterion, str) or not criterion.strip():
            raise InvalidQAError(f"Rubric criterion {index} has no description.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise InvalidQAError(f"Rubric criterion {index} has an invalid weight.")
        if category == "positive" and weight > 0:
            positive_count += 1
        elif category == "negative" and weight < 0:
            negative_count += 1
        else:
            raise InvalidQAError(f"Rubric criterion {index} category and weight sign do not match.")
        item["category"] = category

    if positive_count == 0 or negative_count == 0:
        raise InvalidQAError("Rubric must include both positive and negative criteria.")


class AgenticSelfInstruct:
    """Agentic Self-Instruct loop for hard synthetic research QA generation."""

    def __init__(
        self,
        weak_solver_avg_max: float = 0.55,
        strong_solver_avg_min: float = 0.65,
        gap_min: float = 0.15,
        max_rounds: int = 10,
        weak_rollouts: int = 3,
        strong_rollouts: int = 3,
        save_dir: str = "data/trajectories",
        accepted_dir: str = "data/accepted",
        quality_check: bool = True,
        budget_usd: Optional[float] = None,
        budget_safety_margin_usd: Optional[float] = None,
        cost_report_path: Optional[str] = None,
        resume_cost_report: bool = True,
    ):
        self._validate_configuration(
            weak_solver_avg_max,
            strong_solver_avg_min,
            gap_min,
            max_rounds,
            weak_rollouts,
            strong_rollouts,
        )
        configure_cost_tracker(
            budget_usd=budget_usd,
            safety_margin_usd=budget_safety_margin_usd,
            report_path=cost_report_path,
            resume=resume_cost_report,
        )

        self.weak_solver_avg_max = weak_solver_avg_max
        self.strong_solver_avg_min = strong_solver_avg_min
        self.gap_min = gap_min
        self.max_rounds = max_rounds
        self.weak_rollouts = weak_rollouts
        self.strong_rollouts = strong_rollouts
        self.quality_check_enabled = quality_check
        self.cost_tracker = get_cost_tracker()

        self.save_dir = Path(save_dir)
        self.accepted_dir = Path(accepted_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_dir.mkdir(parents=True, exist_ok=True)

        self.challenger = ChallengerAgent()
        self.weak_solver = WeakSolverAgent()
        self.strong_solver = StrongSolverAgent()
        self.judge = JudgeAgent()

    def run(self, paper_text: str, paper_id: str = "unknown") -> Optional[Dict]:
        if not paper_text or not paper_text.strip():
            raise ValueError("Paper text is empty.")

        print(f"\n{'=' * 60}")
        print(f"Processing: {paper_id}")
        print(f"{'=' * 60}")

        trajectory = {
            "paper_id": paper_id,
            "rounds": [],
            "accepted": False,
            "final_qa": None,
            "settings": {
                "weak_solver_avg_max": self.weak_solver_avg_max,
                "strong_solver_avg_min": self.strong_solver_avg_min,
                "gap_min": self.gap_min,
                "max_rounds": self.max_rounds,
                "weak_rollouts": self.weak_rollouts,
                "strong_rollouts": self.strong_rollouts,
                "quality_check": self.quality_check_enabled,
                "budget_usd": self.cost_tracker.budget_usd,
                "budget_safety_margin_usd": self.cost_tracker.safety_margin_usd,
            },
        }
        feedback_history: List[str] = []

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num}/{self.max_rounds} ---")

            try:
                qa = self.challenger.generate_qa(paper_text, feedback_history)
                validate_qa_item(qa)
                print(f"Challenger generated: {qa['question_type']}")
            except BudgetExceededError as exc:
                self._stop_for_budget(trajectory, paper_id, round_num, exc)
            except OpenAIError as exc:
                self._stop_for_api_error(trajectory, paper_id, round_num, "CHALLENGER_API_FAILED", exc)
            except (InvalidQAError, ValueError, TypeError) as exc:
                feedback = f"Invalid challenger output: {exc}"
                print(feedback)
                feedback_history.append(f"Round {round_num}: {feedback}")
                trajectory["rounds"].append(
                    {"round": round_num, "status": "INVALID_QA", "error": str(exc)}
                )
                continue

            rubric = qa["rubric"]
            if self.quality_check_enabled:
                try:
                    quality = self.judge.quality_check(qa)
                    if not isinstance(quality, dict) or quality.get("passed") is not True:
                        feedback = (
                            quality.get("feedback", "Quality verifier rejected this item.")
                            if isinstance(quality, dict)
                            else "Quality verifier returned an invalid result."
                        )
                        print(f"Quality check failed: {feedback}")
                        feedback_history.append(f"Round {round_num}: quality check failed. {feedback}")
                        trajectory["rounds"].append(
                            {"round": round_num, "status": "QUALITY_FAILED", "quality": quality}
                        )
                        continue
                except BudgetExceededError as exc:
                    self._stop_for_budget(trajectory, paper_id, round_num, exc)
                except OpenAIError as exc:
                    self._stop_for_api_error(trajectory, paper_id, round_num, "QUALITY_API_FAILED", exc)
                except Exception as exc:
                    feedback = f"Quality verifier errored: {exc}"
                    print(feedback)
                    feedback_history.append(f"Round {round_num}: {feedback}")
                    trajectory["rounds"].append(
                        {"round": round_num, "status": "QUALITY_CHECK_ERROR", "error": str(exc)}
                    )
                    continue

            question = qa["question"]
            context = qa["context"]
            reference_answer = qa["reference_answer"]

            print(f"Running weak solver ({self.weak_rollouts} rollouts)...")
            try:
                weak_scores, weak_answers = self._run_solver_rollouts(
                    solver=self.weak_solver,
                    question=question,
                    context=context,
                    reference_answer=reference_answer,
                    rubric=rubric,
                    rollouts=self.weak_rollouts,
                    label="Weak",
                )
            except BudgetExceededError as exc:
                self._stop_for_budget(trajectory, paper_id, round_num, exc)
            except OpenAIError as exc:
                self._stop_for_api_error(trajectory, paper_id, round_num, "WEAK_API_FAILED", exc)
            except RolloutFailedError as exc:
                feedback = str(exc)
                print(feedback)
                feedback_history.append(f"Round {round_num}: {feedback}")
                trajectory["rounds"].append(
                    {"round": round_num, "qa": qa, "status": "WEAK_ROLLOUT_FAILED", "error": feedback}
                )
                continue

            weak_avg = sum(weak_scores) / len(weak_scores)
            print(f"   Weak scores: {[f'{score:.2f}' for score in weak_scores]}, avg: {weak_avg:.3f}")

            if weak_avg > self.weak_solver_avg_max:
                feedback = f"Too easy. Weak scored {weak_avg:.2f}; make the question harder and less pattern-matchable."
                print(feedback)
                feedback_history.append(f"Round {round_num}: {feedback}")
                trajectory["rounds"].append(
                    {
                        "round": round_num,
                        "qa": qa,
                        "weak_avg": weak_avg,
                        "weak_answers": weak_answers,
                        "status": "TOO_EASY",
                    }
                )
                continue

            print(f"Running strong solver ({self.strong_rollouts} rollouts)...")
            try:
                strong_scores, strong_answers = self._run_solver_rollouts(
                    solver=self.strong_solver,
                    question=question,
                    context=context,
                    reference_answer=reference_answer,
                    rubric=rubric,
                    rollouts=self.strong_rollouts,
                    label="Strong",
                )
            except BudgetExceededError as exc:
                self._stop_for_budget(trajectory, paper_id, round_num, exc)
            except OpenAIError as exc:
                self._stop_for_api_error(trajectory, paper_id, round_num, "STRONG_API_FAILED", exc)
            except RolloutFailedError as exc:
                feedback = str(exc)
                print(feedback)
                feedback_history.append(f"Round {round_num}: {feedback}")
                trajectory["rounds"].append(
                    {
                        "round": round_num,
                        "qa": qa,
                        "weak_avg": weak_avg,
                        "weak_answers": weak_answers,
                        "status": "STRONG_ROLLOUT_FAILED",
                        "error": feedback,
                    }
                )
                continue

            strong_avg = sum(strong_scores) / len(strong_scores)
            gap = strong_avg - weak_avg
            print(f"   Strong scores: {[f'{score:.2f}' for score in strong_scores]}, avg: {strong_avg:.3f}")
            print(f"   Gap: {gap:.3f}")

            round_record = {
                "round": round_num,
                "qa": qa,
                "weak_avg": weak_avg,
                "strong_avg": strong_avg,
                "gap": gap,
                "weak_answers": weak_answers,
                "strong_answers": strong_answers,
            }

            if weak_avg <= self.weak_solver_avg_max and strong_avg >= self.strong_solver_avg_min and gap >= self.gap_min:
                print(f"ACCEPTED! Weak={weak_avg:.3f}, Strong={strong_avg:.3f}, Gap={gap:.3f}")
                qa.update(
                    {
                        "weak_avg": weak_avg,
                        "strong_avg": strong_avg,
                        "gap": gap,
                        "weak_answers": weak_answers,
                        "strong_answers": strong_answers,
                    }
                )
                round_record["status"] = "ACCEPTED"
                trajectory["rounds"].append(round_record)
                trajectory["accepted"] = True
                trajectory["final_qa"] = qa
                self._save_trajectory(trajectory, paper_id)
                self._save_accepted(qa, paper_id)
                return qa

            if strong_avg < self.strong_solver_avg_min:
                feedback = f"Strong solver too low ({strong_avg:.2f}); clarify context, reference answer, or rubric."
            elif gap < self.gap_min:
                feedback = f"Gap too small ({gap:.2f}); make it harder for weak solver but still solvable."
            else:
                feedback = f"Rejected. Weak={weak_avg:.2f}, Strong={strong_avg:.2f}, Gap={gap:.2f}."

            print(f"Rejected: {feedback}")
            feedback_history.append(f"Round {round_num}: {feedback}")
            round_record["status"] = "REJECTED"
            trajectory["rounds"].append(round_record)

        print(f"No item accepted after {self.max_rounds} rounds")
        self._save_trajectory(trajectory, paper_id)
        return None

    def _run_solver_rollouts(
        self,
        solver,
        question: str,
        context: str,
        reference_answer: str,
        rubric: list,
        rollouts: int,
        label: str,
    ):
        scores = []
        answers = []
        for index in range(rollouts):
            try:
                answer = solver.solve(question, context)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("solver returned an empty answer")
                score_result = self.judge.score_answer(
                    question,
                    answer,
                    rubric,
                    reference_answer=reference_answer,
                )
                score = float(score_result.get("normalized_score"))
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"judge returned invalid normalized_score={score!r}")
                answers.append(answer)
                scores.append(score)
            except (BudgetExceededError, OpenAIError):
                raise
            except Exception as exc:
                raise RolloutFailedError(f"{label} rollout {index + 1} failed: {exc}") from exc
        return scores, answers

    def _stop_for_budget(
        self,
        trajectory: Dict,
        paper_id: str,
        round_num: int,
        exc: BudgetExceededError,
    ) -> None:
        print(exc)
        trajectory["rounds"].append({"round": round_num, "status": "BUDGET_STOP", "error": str(exc)})
        trajectory["stopped_reason"] = "budget"
        self._save_trajectory(trajectory, paper_id)
        raise exc

    def _stop_for_api_error(
        self,
        trajectory: Dict,
        paper_id: str,
        round_num: int,
        status: str,
        exc: OpenAIError,
    ) -> None:
        print(f"OpenAI API error: {exc}")
        trajectory["rounds"].append({"round": round_num, "status": status, "error": str(exc)})
        trajectory["stopped_reason"] = "api_error"
        self._save_trajectory(trajectory, paper_id)
        raise exc

    def _save_trajectory(self, trajectory: Dict, paper_id: str) -> None:
        trajectory["cost_summary"] = self.cost_tracker.summary()
        filepath = self.save_dir / f"{paper_id}_trajectory.json"
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(trajectory, file, indent=2, ensure_ascii=False)
        print(f"Saved trajectory: {filepath}")

    def _save_accepted(self, qa: Dict, paper_id: str) -> None:
        filepath = self.accepted_dir / f"{paper_id}_accepted.json"
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(qa, file, indent=2, ensure_ascii=False)

        jsonl_path = self.accepted_dir / "dataset.jsonl"
        record = {
            "paper_id": paper_id,
            "context": qa.get("context", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("reference_answer", ""),
            "rubric": qa.get("rubric", []),
            "weak_avg": qa.get("weak_avg"),
            "strong_avg": qa.get("strong_avg"),
            "gap": qa.get("gap"),
            "question_type": qa.get("question_type"),
            "reasoning_skills": qa.get("reasoning_skills", []),
        }

        retained_lines = []
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    retained_lines.append(line)
                    continue
                if not isinstance(existing, dict) or existing.get("paper_id") != paper_id:
                    retained_lines.append(json.dumps(existing, ensure_ascii=False))

        retained_lines.append(json.dumps(record, ensure_ascii=False))
        temp_path = jsonl_path.with_name(f"{jsonl_path.name}.tmp")
        temp_path.write_text("\n".join(retained_lines) + "\n", encoding="utf-8")
        temp_path.replace(jsonl_path)
        print(f"Saved accepted item: {filepath}")

    @staticmethod
    def _validate_configuration(
        weak_solver_avg_max: float,
        strong_solver_avg_min: float,
        gap_min: float,
        max_rounds: int,
        weak_rollouts: int,
        strong_rollouts: int,
    ) -> None:
        if max_rounds < 1 or weak_rollouts < 1 or strong_rollouts < 1:
            raise ValueError("max_rounds and rollout counts must be at least 1.")
        for name, value in (
            ("weak_solver_avg_max", weak_solver_avg_max),
            ("strong_solver_avg_min", strong_solver_avg_min),
            ("gap_min", gap_min),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


def batch_process(
    papers_dir: str = "data/raw_text",
    max_papers: int = 5,
    max_rounds: int = 8,
    weak_rollouts: int = 2,
    strong_rollouts: int = 2,
    weak_solver_avg_max: float = 0.55,
    strong_solver_avg_min: float = 0.65,
    gap_min: float = 0.15,
    quality_check: bool = True,
    paper_delay: float = 1.0,
    budget_usd: Optional[float] = None,
    budget_safety_margin_usd: Optional[float] = None,
    cost_report_path: Optional[str] = None,
):
    if max_papers < 0:
        raise ValueError("max_papers must be zero or greater.")
    if paper_delay < 0:
        raise ValueError("paper_delay must be zero or greater.")
    papers_path = Path(papers_dir)
    if not papers_path.is_dir():
        raise ValueError(f"Papers directory does not exist: {papers_path}")

    pipeline = AgenticSelfInstruct(
        weak_solver_avg_max=weak_solver_avg_max,
        strong_solver_avg_min=strong_solver_avg_min,
        gap_min=gap_min,
        max_rounds=max_rounds,
        weak_rollouts=weak_rollouts,
        strong_rollouts=strong_rollouts,
        quality_check=quality_check,
        budget_usd=budget_usd,
        budget_safety_margin_usd=budget_safety_margin_usd,
        cost_report_path=cost_report_path,
    )

    papers = sorted(papers_path.glob("*.txt"))[:max_papers]
    accepted = []
    rejected = []
    processed = []
    fatal_error: Optional[OpenAIError] = None
    budget_stopped = False

    for paper_path in tqdm(papers, desc="Processing papers"):
        paper_id = paper_path.stem
        try:
            paper_text = paper_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"Could not read {paper_path}: {exc}")
            rejected.append(paper_id)
            processed.append(paper_id)
            continue
        if not paper_text.strip():
            print(f"Skipping empty paper: {paper_path}")
            rejected.append(paper_id)
            processed.append(paper_id)
            continue

        processed.append(paper_id)
        try:
            result = pipeline.run(paper_text, paper_id)
        except BudgetExceededError:
            rejected.append(paper_id)
            budget_stopped = True
            break
        except OpenAIError as exc:
            rejected.append(paper_id)
            fatal_error = exc
            break

        if result:
            accepted.append(paper_id)
        else:
            rejected.append(paper_id)

        if paper_delay > 0:
            time.sleep(paper_delay)

    print(f"\n{'=' * 60}")
    print("BATCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Selected: {len(papers)}")
    print(f"Processed: {len(processed)}")
    rate = (len(accepted) / len(processed) * 100) if processed else 0.0
    print(f"Accepted: {len(accepted)} ({rate:.1f}%)")
    print(f"Rejected: {len(rejected)}")
    print(f"Budget stopped: {budget_stopped}")
    if accepted:
        print(f"Accepted papers: {accepted}")
    print(f"Cost summary: {pipeline.cost_tracker.summary()}")

    if fatal_error is not None:
        raise fatal_error
    return accepted, rejected


if __name__ == "__main__":
    sample_paper = """Title: Attention Is All You Need
Authors: Vaswani et al., 2017

The Transformer replaces recurrence and convolutions with attention mechanisms.
Experiments on WMT 2014 English-to-German and English-to-French established strong translation results."""
    pipeline = AgenticSelfInstruct(max_rounds=3, weak_rollouts=1, strong_rollouts=1)
    result = pipeline.run(sample_paper, "attention_paper")
    print("Accepted." if result else "No acceptance.")
