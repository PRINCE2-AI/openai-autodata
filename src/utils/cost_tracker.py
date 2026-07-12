import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


class BudgetExceededError(RuntimeError):
    pass


MODEL_PRICING_USD_PER_1M = {
    "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.6-luna": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _field(value: Any, name: str, default: Any = 0) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class CostTracker:
    """Persistently tracks conservative API-cost estimates and enforces a guard."""

    def __init__(
        self,
        budget_usd: Optional[float] = None,
        safety_margin_usd: Optional[float] = None,
        report_path: Optional[str] = None,
        resume: bool = True,
        input_safety_multiplier: Optional[float] = None,
    ):
        self.budget_usd = budget_usd if budget_usd is not None else _float_env("OPENAI_BUDGET_USD", 50.0)
        self.safety_margin_usd = (
            safety_margin_usd if safety_margin_usd is not None else _float_env("OPENAI_BUDGET_SAFETY_USD", 2.0)
        )
        self.input_safety_multiplier = (
            input_safety_multiplier
            if input_safety_multiplier is not None
            else _float_env("OPENAI_COST_INPUT_SAFETY_MULTIPLIER", 1.25)
        )
        self.report_path = Path(report_path or os.getenv("OPENAI_COST_REPORT_PATH", "data/cost_report.jsonl"))
        self.session_id = uuid4().hex
        self.spent_usd = 0.0
        self.call_count = 0
        self.by_model: Dict[str, Dict[str, float]] = {}
        self.load_warnings = 0

        self._validate_settings()
        if resume:
            self._load_existing_records()
        self.initial_spent_usd = self.spent_usd
        self.initial_call_count = self.call_count

    @property
    def guard_limit_usd(self) -> float:
        return self.budget_usd - self.safety_margin_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    @property
    def guard_remaining_usd(self) -> float:
        return max(0.0, self.guard_limit_usd - self.spent_usd)

    @property
    def session_spent_usd(self) -> float:
        return max(0.0, self.spent_usd - self.initial_spent_usd)

    def ensure_can_call(
        self,
        model: str,
        estimated_input_tokens: int = 0,
        max_output_tokens: int = 0,
    ) -> float:
        if estimated_input_tokens < 0 or max_output_tokens < 0:
            raise ValueError("Estimated input and output tokens must be non-negative.")

        estimated_next_cost = self.estimate_cost(
            model,
            {
                "prompt_tokens": estimated_input_tokens,
                "cached_tokens": 0,
                "completion_tokens": max_output_tokens,
            },
        )
        projected_spend = self.spent_usd + estimated_next_cost
        if self.spent_usd >= self.guard_limit_usd or projected_spend > self.guard_limit_usd:
            raise BudgetExceededError(
                f"Budget guard stopped before calling {model}. "
                f"Estimated spent=${self.spent_usd:.4f}, next call up to=${estimated_next_cost:.4f}, "
                f"guard limit=${self.guard_limit_usd:.2f}, total budget=${self.budget_usd:.2f}."
            )
        return estimated_next_cost

    def record_response(
        self,
        model: str,
        usage: Any,
        role: str = "",
        fallback_tokens: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        tokens = self._extract_tokens(usage)
        usage_estimated = usage is None or not any(tokens.values())
        if usage_estimated and fallback_tokens:
            tokens = {
                "prompt_tokens": int(fallback_tokens.get("prompt_tokens", 0)),
                "cached_tokens": int(fallback_tokens.get("cached_tokens", 0)),
                "completion_tokens": int(fallback_tokens.get("completion_tokens", 0)),
            }

        cost = self.estimate_cost(model, tokens)
        self._add_to_totals(model, tokens, cost)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "role": role,
            "model": model,
            "prompt_tokens": tokens["prompt_tokens"],
            "cached_tokens": tokens["cached_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "usage_estimated": usage_estimated,
            "cost_usd": round(cost, 8),
            "spent_usd": round(self.spent_usd, 8),
            "remaining_usd": round(self.remaining_usd, 8),
            "guard_remaining_usd": round(self.guard_remaining_usd, 8),
        }
        self._append_record(record)
        return record

    def estimate_cost(self, model: str, tokens: Dict[str, int]) -> float:
        pricing = self._pricing_for_model(model)
        prompt_tokens = max(int(tokens.get("prompt_tokens", 0)) - int(tokens.get("cached_tokens", 0)), 0)
        cached_tokens = max(int(tokens.get("cached_tokens", 0)), 0)
        completion_tokens = max(int(tokens.get("completion_tokens", 0)), 0)
        return (
            prompt_tokens * pricing["input"] * self.input_safety_multiplier
            + cached_tokens * pricing["cached_input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

    def summary(self) -> Dict[str, Any]:
        return {
            "budget_usd": self.budget_usd,
            "safety_margin_usd": self.safety_margin_usd,
            "guard_limit_usd": round(self.guard_limit_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "session_spent_usd": round(self.session_spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "guard_remaining_usd": round(self.guard_remaining_usd, 6),
            "call_count": self.call_count,
            "session_call_count": self.call_count - self.initial_call_count,
            "resumed_from_report": self.initial_call_count > 0,
            "load_warnings": self.load_warnings,
            "by_model": self.by_model,
        }

    def _validate_settings(self) -> None:
        values = (self.budget_usd, self.safety_margin_usd, self.input_safety_multiplier)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Budget settings must be finite numbers.")
        if self.budget_usd <= 0:
            raise ValueError("OPENAI_BUDGET_USD must be greater than zero.")
        if self.safety_margin_usd < 0 or self.safety_margin_usd >= self.budget_usd:
            raise ValueError("Budget safety margin must be non-negative and smaller than the total budget.")
        if self.input_safety_multiplier < 1:
            raise ValueError("OPENAI_COST_INPUT_SAFETY_MULTIPLIER must be at least 1.0.")

    def _pricing_for_model(self, model: str) -> Dict[str, float]:
        if model in MODEL_PRICING_USD_PER_1M:
            return MODEL_PRICING_USD_PER_1M[model]
        for model_id in sorted(MODEL_PRICING_USD_PER_1M, key=len, reverse=True):
            if model.startswith(f"{model_id}-"):
                return MODEL_PRICING_USD_PER_1M[model_id]
        return self._fallback_pricing()

    def _add_to_totals(self, model: str, tokens: Dict[str, int], cost: float) -> None:
        self.spent_usd += cost
        self.call_count += 1
        model_stats = self.by_model.setdefault(
            model,
            {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )
        model_stats["calls"] += 1
        model_stats["prompt_tokens"] += tokens["prompt_tokens"]
        model_stats["cached_tokens"] += tokens["cached_tokens"]
        model_stats["completion_tokens"] += tokens["completion_tokens"]
        model_stats["cost_usd"] += cost

    def _load_existing_records(self) -> None:
        if not self.report_path.exists():
            return
        try:
            lines = self.report_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"Could not read existing cost report {self.report_path}: {exc}") from exc

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                model = str(record["model"])
                tokens = {
                    "prompt_tokens": max(int(record.get("prompt_tokens", 0)), 0),
                    "cached_tokens": max(int(record.get("cached_tokens", 0)), 0),
                    "completion_tokens": max(int(record.get("completion_tokens", 0)), 0),
                }
                cost = max(float(record["cost_usd"]), 0.0)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.load_warnings += 1
                continue
            self._add_to_totals(model, tokens, cost)

    def _append_record(self, record: Dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_tokens(usage: Any) -> Dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}

        prompt_tokens = int(_field(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_field(usage, "completion_tokens", 0) or 0)
        details = _field(usage, "prompt_tokens_details", None)
        cached_tokens = int(_field(details, "cached_tokens", 0) or 0) if details is not None else 0
        cached_tokens = min(max(cached_tokens, 0), max(prompt_tokens, 0))
        return {
            "prompt_tokens": max(prompt_tokens, 0),
            "cached_tokens": cached_tokens,
            "completion_tokens": max(completion_tokens, 0),
        }

    @staticmethod
    def _fallback_pricing() -> Dict[str, float]:
        return {
            "input": _float_env("OPENAI_DEFAULT_INPUT_PER_1M", 5.0),
            "cached_input": _float_env("OPENAI_DEFAULT_CACHED_INPUT_PER_1M", 0.5),
            "output": _float_env("OPENAI_DEFAULT_OUTPUT_PER_1M", 30.0),
        }


_TRACKER: Optional[CostTracker] = None


def configure_cost_tracker(
    budget_usd: Optional[float] = None,
    safety_margin_usd: Optional[float] = None,
    report_path: Optional[str] = None,
    resume: bool = True,
) -> CostTracker:
    global _TRACKER
    _TRACKER = CostTracker(
        budget_usd=budget_usd,
        safety_margin_usd=safety_margin_usd,
        report_path=report_path,
        resume=resume,
    )
    return _TRACKER


def get_cost_tracker() -> CostTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = CostTracker()
    return _TRACKER
