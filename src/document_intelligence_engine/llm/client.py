"""LLM client with prompt caching, JSON parsing, and safe mock fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


class LLMClient:
    """Small experiment-oriented LLM client with persistent on-disk caching."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        model: str = "mock-llm",
        cache_dir: str | Path = "experiments/cache/llm",
        base_url: str | None = None,
        pricing: dict[str, float] | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pricing = pricing or {"input_per_1k": 0.0015, "output_per_1k": 0.0020}
        self.total_cost = 0.0

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "backend": self.backend,
                    "model": self.model,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cache_hit"] = True
            return payload

        if self.backend == "mock":
            payload = self._call_mock(prompt)
        elif self.backend in {"openai", "openai_compatible"}:
            payload = self._call_openai(prompt, system_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.total_cost += float(payload["cost_usd"])
        payload["cache_hit"] = False
        return payload

    def extract_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = self.generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_text = str(payload["text"]).strip()
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            repaired = self._repair_json(raw_text)
            parsed = json.loads(repaired)
        return {**payload, "parsed": parsed}

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("The 'openai' package is required for backend='openai'.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
        return {
            "text": text.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }

    def _call_mock(self, prompt: str) -> dict[str, Any]:
        text = _mock_response(prompt)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(text)
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": 0.0,
        }

    def _repair_json(self, raw_text: str) -> str:
        try:
            from json_repair import repair_json
        except ImportError:
            candidate = _extract_json_candidate(raw_text)
            if candidate is None:
                raise ValueError("Could not repair invalid JSON output.")
            return candidate
        return repair_json(raw_text)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1000.0) * float(self.pricing["input_per_1k"])
        output_cost = (completion_tokens / 1000.0) * float(self.pricing["output_per_1k"])
        return round(input_cost + output_cost, 6)


def _extract_json_candidate(raw_text: str) -> str | None:
    starts = [index for index in (raw_text.find("{"), raw_text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    end_candidates = [raw_text.rfind("}"), raw_text.rfind("]")]
    end = max(candidate for candidate in end_candidates if candidate >= 0)
    if end < start:
        return None
    return raw_text[start : end + 1]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _mock_response(prompt: str) -> str:
    if 'field "' in prompt:
        field_match = re.search(r'field "([^"]+)"', prompt)
        field_name = field_match.group(1) if field_match else ""
        context = prompt.split("Context:", maxsplit=1)[-1]
        extracted = _heuristic_extract_fields(context)
        return json.dumps(extracted.get(field_name))

    source_text = prompt.split("OCR Text:", maxsplit=1)[-1]
    extracted = _heuristic_extract_fields(source_text)
    return json.dumps(extracted)


def _heuristic_extract_fields(source_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    invoice_match = re.search(
        r"(?:invoice(?:\s+(?:number|no))?|receipt)\s*#?\s*[:\-]?\s*([A-Za-z0-9/_-]+)",
        joined,
        flags=re.IGNORECASE,
    )
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", joined)
    vendor_match = re.search(r"vendor\s*[:\-]?\s*(.+)", joined, flags=re.IGNORECASE)
    total_match = re.search(
        r"(?:total(?:\s+amount)?|amount\s+due)\s*[:\-]?\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)",
        joined,
        flags=re.IGNORECASE,
    )

    line_items: list[dict[str, Any]] = []
    for line in lines:
        line_match = re.search(
            r"item\s*[:\-]?\s*([^,]+),\s*qty\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)"
            r",\s*price\s*[:\-]?\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)",
            line,
            flags=re.IGNORECASE,
        )
        if line_match:
            line_items.append(
                {
                    "description": line_match.group(1).strip(),
                    "quantity": float(line_match.group(2)),
                    "unit_price": float(line_match.group(3)),
                }
            )

    return {
        "invoice_number": invoice_match.group(1) if invoice_match else None,
        "date": date_match.group(1) if date_match else None,
        "vendor": vendor_match.group(1).strip() if vendor_match else None,
        "total_amount": float(total_match.group(1)) if total_match else None,
        "line_items": line_items,
    }
