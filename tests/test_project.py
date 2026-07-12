import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agents.openai_agent import OpenAIAgent
from src.pipeline.agentic_loop import (
    AgenticSelfInstruct,
    InvalidQAError,
    RolloutFailedError,
    batch_process,
    validate_qa_item,
)
from src.utils.cost_tracker import BudgetExceededError, CostTracker, configure_cost_tracker


def valid_qa() -> dict:
    rubric = [
        {"criterion": f"Positive criterion {index}", "weight": 2, "category": "positive"}
        for index in range(9)
    ]
    rubric.append({"criterion": "Penalize a factual error", "weight": -3, "category": "negative"})
    return {
        "question_type": "failure mode prediction",
        "reasoning_skills": ["causal reasoning", "experimental design"],
        "context": "Enough context to solve the problem without leaking the answer.",
        "question": "What failure occurs under the stated intervention, and how should it be tested?",
        "reference_answer": "The intervention changes the bottleneck; test it with a controlled ablation.",
        "rubric": rubric,
    }


class CostTrackerTests(unittest.TestCase):
    def test_cost_estimate_handles_cached_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CostTracker(
                budget_usd=10,
                safety_margin_usd=1,
                report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume=False,
                input_safety_multiplier=1.0,
            )
            cost = tracker.estimate_cost(
                "gpt-5.6-luna",
                {"prompt_tokens": 1000, "cached_tokens": 200, "completion_tokens": 500},
            )
            self.assertAlmostEqual(cost, 0.00382)

    def test_preflight_blocks_a_call_that_would_cross_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CostTracker(
                budget_usd=1,
                safety_margin_usd=0.1,
                report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume=False,
                input_safety_multiplier=1.0,
            )
            tracker.spent_usd = 0.89
            with self.assertRaises(BudgetExceededError):
                tracker.ensure_can_call("gpt-5.6-sol", max_output_tokens=1000)

    def test_existing_ledger_is_resumed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = str(Path(temp_dir) / "cost.jsonl")
            first = CostTracker(
                budget_usd=10,
                safety_margin_usd=1,
                report_path=report_path,
                resume=False,
                input_safety_multiplier=1.0,
            )
            first.record_response(
                "gpt-5.6-luna",
                {"prompt_tokens": 1000, "completion_tokens": 500},
                role="test",
            )

            resumed = CostTracker(
                budget_usd=10,
                safety_margin_usd=1,
                report_path=report_path,
                resume=True,
                input_safety_multiplier=1.0,
            )
            self.assertEqual(resumed.call_count, 1)
            self.assertAlmostEqual(resumed.spent_usd, first.spent_usd)
            self.assertEqual(resumed.session_spent_usd, 0)


class OpenAIAgentTests(unittest.TestCase):
    def test_json_mode_and_usage_tracking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_cost_tracker(
                budget_usd=10,
                safety_margin_usd=1,
                report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume=False,
            )
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )
            client = MagicMock()
            client.chat.completions.create.return_value = response

            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-test", "OPENAI_TIMEOUT_SECONDS": "1"},
                clear=False,
            ), patch("src.agents.openai_agent.OpenAI", return_value=client):
                agent = OpenAIAgent(
                    model_name="gpt-5.6-luna",
                    temperature=None,
                    max_tokens=100,
                    reasoning_effort="low",
                    min_delay=0,
                )
                result = agent.generate_json("system", "user")

            self.assertEqual(result, {"ok": True})
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})
            self.assertEqual(kwargs["reasoning_effort"], "low")
            self.assertEqual(kwargs["max_completion_tokens"], 100)
            self.assertEqual(len(Path(temp_dir, "cost.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_budget_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configure_cost_tracker(
                budget_usd=0.01,
                safety_margin_usd=0.001,
                report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume=False,
            )
            client = MagicMock()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
                "src.agents.openai_agent.OpenAI", return_value=client
            ):
                agent = OpenAIAgent(
                    model_name="gpt-5.6-sol",
                    temperature=None,
                    max_tokens=100_000,
                    min_delay=0,
                )
                with self.assertRaises(BudgetExceededError):
                    agent.generate("", "hello")
            client.chat.completions.create.assert_not_called()


class PipelineTests(unittest.TestCase):
    def test_qa_validation_rejects_bad_weight_sign(self):
        qa = valid_qa()
        qa["rubric"][-1]["weight"] = 3
        with self.assertRaises(InvalidQAError):
            validate_qa_item(qa)

    def test_failed_rollout_is_not_scored_as_zero(self):
        pipeline = AgenticSelfInstruct.__new__(AgenticSelfInstruct)
        pipeline.judge = SimpleNamespace(score_answer=MagicMock(side_effect=ValueError("bad judge output")))
        solver = SimpleNamespace(solve=MagicMock(return_value="answer"))
        with self.assertRaises(RolloutFailedError):
            pipeline._run_solver_rollouts(
                solver=solver,
                question="question",
                context="context",
                reference_answer="reference",
                rubric=valid_qa()["rubric"],
                rollouts=1,
                label="Weak",
            )

    def test_reference_answer_is_given_to_judge(self):
        judge = MagicMock()
        judge.score_answer.return_value = {"normalized_score": 0.5}
        pipeline = AgenticSelfInstruct.__new__(AgenticSelfInstruct)
        pipeline.judge = judge
        solver = SimpleNamespace(solve=MagicMock(return_value="candidate"))

        scores, _ = pipeline._run_solver_rollouts(
            solver=solver,
            question="question",
            context="context",
            reference_answer="gold answer",
            rubric=valid_qa()["rubric"],
            rollouts=1,
            label="Strong",
        )

        self.assertEqual(scores, [0.5])
        self.assertEqual(judge.score_answer.call_args.kwargs["reference_answer"], "gold answer")

    def test_budget_stop_saves_trajectory(self):
        class BudgetChallenger:
            def generate_qa(self, paper_text, feedback_history):
                raise BudgetExceededError("budget reached")

        class DummyAgent:
            pass

        with tempfile.TemporaryDirectory() as temp_dir, patch.multiple(
            "src.pipeline.agentic_loop",
            ChallengerAgent=BudgetChallenger,
            WeakSolverAgent=DummyAgent,
            StrongSolverAgent=DummyAgent,
            JudgeAgent=DummyAgent,
        ):
            pipeline = AgenticSelfInstruct(
                max_rounds=1,
                weak_rollouts=1,
                strong_rollouts=1,
                save_dir=str(Path(temp_dir) / "trajectories"),
                accepted_dir=str(Path(temp_dir) / "accepted"),
                cost_report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume_cost_report=False,
            )
            with self.assertRaises(BudgetExceededError):
                pipeline.run("paper text", "paper-1")

            trajectory_path = Path(temp_dir, "trajectories", "paper-1_trajectory.json")
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(trajectory["rounds"][-1]["status"], "BUDGET_STOP")
            self.assertEqual(trajectory["stopped_reason"], "budget")

    def test_end_to_end_fake_agents_accept_and_save_item(self):
        class GoodChallenger:
            def generate_qa(self, paper_text, feedback_history):
                return valid_qa()

        class WeakSolver:
            def solve(self, question, context):
                return "weak answer"

        class StrongSolver:
            def solve(self, question, context):
                return "strong answer"

        class Judge:
            def quality_check(self, qa):
                return {"passed": True, "issues": [], "feedback": "ok"}

            def score_answer(self, question, answer, rubric, reference_answer=""):
                return {"normalized_score": 0.2 if answer == "weak answer" else 0.8}

        with tempfile.TemporaryDirectory() as temp_dir, patch.multiple(
            "src.pipeline.agentic_loop",
            ChallengerAgent=GoodChallenger,
            WeakSolverAgent=WeakSolver,
            StrongSolverAgent=StrongSolver,
            JudgeAgent=Judge,
        ):
            pipeline = AgenticSelfInstruct(
                max_rounds=1,
                weak_rollouts=1,
                strong_rollouts=1,
                save_dir=str(Path(temp_dir) / "trajectories"),
                accepted_dir=str(Path(temp_dir) / "accepted"),
                cost_report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume_cost_report=False,
            )
            result = pipeline.run("paper text", "paper-accepted")

            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["gap"], 0.6)
            self.assertTrue(Path(temp_dir, "accepted", "paper-accepted_accepted.json").exists())
            dataset = Path(temp_dir, "accepted", "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(dataset), 1)

    def test_string_false_does_not_pass_quality_check(self):
        class GoodChallenger:
            def generate_qa(self, paper_text, feedback_history):
                return valid_qa()

        class SolverMustNotRun:
            def solve(self, question, context):
                raise AssertionError("solver should not run after a failed quality check")

        class StringFalseJudge:
            def quality_check(self, qa):
                return {"passed": "false", "feedback": "not actually a boolean"}

        with tempfile.TemporaryDirectory() as temp_dir, patch.multiple(
            "src.pipeline.agentic_loop",
            ChallengerAgent=GoodChallenger,
            WeakSolverAgent=SolverMustNotRun,
            StrongSolverAgent=SolverMustNotRun,
            JudgeAgent=StringFalseJudge,
        ):
            pipeline = AgenticSelfInstruct(
                max_rounds=1,
                weak_rollouts=1,
                strong_rollouts=1,
                save_dir=str(Path(temp_dir) / "trajectories"),
                accepted_dir=str(Path(temp_dir) / "accepted"),
                cost_report_path=str(Path(temp_dir) / "cost.jsonl"),
                resume_cost_report=False,
            )
            result = pipeline.run("paper text", "paper-quality-failed")

            self.assertIsNone(result)
            trajectory = json.loads(
                Path(temp_dir, "trajectories", "paper-quality-failed_trajectory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trajectory["rounds"][-1]["status"], "QUALITY_FAILED")

    def test_dataset_jsonl_is_upserted_by_paper_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = AgenticSelfInstruct.__new__(AgenticSelfInstruct)
            pipeline.accepted_dir = Path(temp_dir)
            first = valid_qa()
            first.update({"weak_avg": 0.2, "strong_avg": 0.8, "gap": 0.6})
            second = valid_qa()
            second["question"] = "updated question"
            second.update({"weak_avg": 0.1, "strong_avg": 0.9, "gap": 0.8})

            pipeline._save_accepted(first, "same-paper")
            pipeline._save_accepted(second, "same-paper")

            lines = Path(temp_dir, "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["question"], "updated question")

    def test_negative_max_papers_is_rejected_before_agent_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                batch_process(papers_dir=temp_dir, max_papers=-1)


if __name__ == "__main__":
    unittest.main()
