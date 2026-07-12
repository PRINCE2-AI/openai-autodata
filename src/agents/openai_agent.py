import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.cost_tracker import get_cost_tracker

load_dotenv()


def _is_retryable_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)


class OpenAIAgent:
    """Base OpenAI agent with retries, light rate limiting, and JSON repair."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        reasoning_effort: Optional[str] = None,
        min_delay: float = 0.2,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        placeholder_keys = {"your-openai-api-key"}
        if not api_key or api_key.strip() in placeholder_keys:
            raise ValueError("OPENAI_API_KEY not found. Add it to .env or your shell environment.")

        timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
        if timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero.")

        # Tenacity owns the retry policy so non-transient API errors are never retried.
        self.client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout_seconds)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.min_delay = min_delay
        self.last_request_time = 0.0
        self.cost_tracker = get_cost_tracker()

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    @retry(
        retry=retry_if_exception(_is_retryable_openai_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        estimated_input_tokens = self._estimate_input_tokens(messages)
        self.cost_tracker.ensure_can_call(
            self.model_name,
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=self.max_tokens,
        )
        self._rate_limit()

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        record = self.cost_tracker.record_response(
            model=self.model_name,
            usage=getattr(response, "usage", None),
            role=self.__class__.__name__,
            fallback_tokens={
                "prompt_tokens": estimated_input_tokens,
                "cached_tokens": 0,
                "completion_tokens": self.max_tokens,
            },
        )
        print(
            f"   Cost: {self.__class__.__name__} {self.model_name} "
            f"${record['cost_usd']:.5f} | total ${record['spent_usd']:.4f} "
            f"| guard remaining ${record['guard_remaining_usd']:.2f}"
        )
        if not getattr(response, "choices", None):
            raise ValueError("OpenAI returned no completion choices.")
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError("OpenAI returned an empty completion.")
        return content

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.generate(
            system_prompt,
            user_prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown fences or explanations.",
            json_mode=True,
        )
        return self._parse_json(response)

    @staticmethod
    def _estimate_input_tokens(messages: List[Dict[str, str]]) -> int:
        # Conservative character estimate used only for the pre-call budget guard.
        content_chars = sum(len(message.get("content", "")) for message in messages)
        return max(1, math.ceil(content_chars / 3) + 12 * len(messages))

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = raw.strip()
        for pattern in (r"^```json\s*", r"^```\s*", r"\s*```$", r"^json\s*"):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start_obj = cleaned.find("{")
            start_arr = cleaned.find("[")
            starts = [idx for idx in (start_obj, start_arr) if idx != -1]
            if not starts:
                raise ValueError(f"No JSON found in model response: {cleaned[:300]}")
            start = min(starts)
            end = cleaned.rfind("}") + 1 if cleaned[start] == "{" else cleaned.rfind("]") + 1
            if end <= start:
                raise ValueError(f"Invalid JSON structure in model response: {cleaned[:300]}")
            parsed = json.loads(cleaned[start:end])

        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")
        return parsed


def env_model(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


class ChallengerAgent(OpenAIAgent):
    def __init__(self):
        super().__init__(
            model_name=env_model("OPENAI_CHALLENGER_MODEL", "gpt-5.6-luna"),
            temperature=float(os.getenv("OPENAI_CHALLENGER_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_CHALLENGER_MAX_TOKENS", "1800")),
            reasoning_effort=os.getenv("OPENAI_CHALLENGER_REASONING", "medium"),
            min_delay=float(os.getenv("OPENAI_MIN_DELAY", "0.4")),
        )

    def generate_qa(self, paper_text: str, feedback_history: Optional[List[str]] = None) -> dict:
        system = (
            "You are an expert computer science researcher building hard synthetic "
            "training examples in the style of Agentic Self-Instruct."
        )
        feedback_str = ""
        if feedback_history:
            feedback_str = "\n\nPrevious failed attempts:\n" + "\n".join(feedback_history[-5:])

        user = f"""Generate ONE challenging research question-answer pair with a detailed grading rubric from this paper.

PAPER:
{paper_text[:9000]}

{feedback_str}

Hard-mode requirements:
- The question must require multi-step reasoning, tradeoff analysis, failure-mode prediction, or experimental design.
- The context must be useful but must NOT leak the final answer.
- Avoid shallow questions like "explain the method" or "summarize the paper".
- The weak solver should plausibly fail; the strong solver should be able to succeed from the context.
- Rubric must contain 10-15 criteria with positive and negative weights.

Return ONLY valid JSON in this exact format:
{{
  "question_type": "short phrase",
  "reasoning_skills": ["skill1", "skill2"],
  "context": "2-3 paragraphs",
  "question": "one hard question",
  "reference_answer": "detailed correct answer",
  "rubric": [
    {{"criterion": "specific grading criterion", "weight": 5, "category": "positive"}},
    {{"criterion": "specific penalty criterion", "weight": -3, "category": "negative"}}
  ]
}}"""
        return self.generate_json(system, user)


class WeakSolverAgent(OpenAIAgent):
    def __init__(self):
        super().__init__(
            model_name=env_model("OPENAI_WEAK_MODEL", "gpt-5.4-nano"),
            temperature=float(os.getenv("OPENAI_WEAK_TEMPERATURE", "1.0")),
            max_tokens=int(os.getenv("OPENAI_WEAK_MAX_TOKENS", "180")),
            reasoning_effort=os.getenv("OPENAI_WEAK_REASONING", "low"),
            min_delay=float(os.getenv("OPENAI_MIN_DELAY", "0.4")),
        )

    def solve(self, question: str, context: str = "") -> str:
        user = f"""Answer this research question in 1-2 sentences only. Do not show detailed reasoning.

Context:
{context[:2200]}

Question:
{question}

Answer:"""
        return self.generate("", user)


class StrongSolverAgent(OpenAIAgent):
    def __init__(self):
        super().__init__(
            model_name=env_model("OPENAI_STRONG_MODEL", "gpt-5.6-luna"),
            temperature=float(os.getenv("OPENAI_STRONG_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("OPENAI_STRONG_MAX_TOKENS", "1000")),
            reasoning_effort=os.getenv("OPENAI_STRONG_REASONING", "high"),
            min_delay=float(os.getenv("OPENAI_MIN_DELAY", "0.4")),
        )

    def solve(self, question: str, context: str = "") -> str:
        user = f"""Think carefully and answer the research question. Use the context, make assumptions explicit, and cover edge cases.

Context:
{context[:2600]}

Question:
{question}

Detailed answer:"""
        return self.generate("", user)


class JudgeAgent(OpenAIAgent):
    def __init__(self):
        super().__init__(
            model_name=env_model("OPENAI_JUDGE_MODEL", "gpt-5.6-luna"),
            temperature=float(os.getenv("OPENAI_JUDGE_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_JUDGE_MAX_TOKENS", "900")),
            reasoning_effort=os.getenv("OPENAI_JUDGE_REASONING", "medium"),
            min_delay=float(os.getenv("OPENAI_MIN_DELAY", "0.4")),
        )

    def score_answer(
        self,
        question: str,
        answer: str,
        rubric: list,
        reference_answer: str = "",
    ) -> dict:
        system = "You are a strict evaluator. Apply the rubric literally and return calibrated scores."
        rubric_str = json.dumps(rubric, indent=2, ensure_ascii=False)
        user = f"""Score this answer against the rubric.

Question:
{question}

Reference answer:
{reference_answer[:4000]}

Answer:
{answer[:3000]}

Rubric:
{rubric_str}

For each criterion, score 1 if fully satisfied and 0 if not satisfied. For negative criteria, score 1 when the bad behavior is present.
Compute weighted_score by summing score * weight. Compute max_positive as the sum of positive weights.
normalized_score should be clamped between 0 and 1.

Return JSON:
{{
  "criterion_scores": [0, 1, 0],
  "weighted_score": 12,
  "max_positive": 35,
  "normalized_score": 0.34,
  "feedback": "brief justification"
}}"""
        result = self.generate_json(system, user)
        return normalize_judge_result(result, rubric)

    def quality_check(self, qa: dict) -> dict:
        system = "You are a quality verifier for synthetic training data."
        user = f"""Evaluate whether this generated QA item is high quality.

QA item:
{json.dumps(qa, ensure_ascii=False)[:6000]}

Check:
- context does not leak the answer
- question needs deep reasoning
- reference answer is complete
- rubric is specific and usable

Return JSON:
{{
  "passed": true,
  "issues": ["issue if any"],
  "feedback": "actionable feedback for the next attempt"
}}"""
        return self.generate_json(system, user)


def normalize_judge_result(result: dict, rubric: list) -> dict:
    scores = result.get("criterion_scores", [])
    if not isinstance(scores, list):
        scores = []

    clean_scores = []
    for idx in range(len(rubric)):
        value = scores[idx] if idx < len(scores) else 0
        clean_scores.append(1 if value in (1, True, "1", "true", "yes") else 0)

    weights = [float(item.get("weight", 0)) for item in rubric]
    weighted_score = sum(score * weight for score, weight in zip(clean_scores, weights))
    max_positive = sum(weight for weight in weights if weight > 0) or 1.0
    normalized = max(0.0, min(1.0, weighted_score / max_positive))

    result["criterion_scores"] = clean_scores
    result["weighted_score"] = weighted_score
    result["max_positive"] = max_positive
    result["normalized_score"] = normalized
    return result
