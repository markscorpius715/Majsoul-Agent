from dataclasses import dataclass, field
import asyncio
import json
import math
import os
from pathlib import Path
import time
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from .domain import Action, State
from .policy_prompt import model_instructions


class StrategyError(RuntimeError):
    pass


class ChoicePolicy(Protocol):
    """Backend-independent action selection with a shared monotonic deadline."""
    def choose(self, state: State, actions: list[Action], *, deadline: float | None = None) -> Action: ...


def read_model_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    allowed = {"DEEPSEEK_API_KEY", "API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_TIMEOUT_SECONDS", "DEEPSEEK_THINKING", "DEEPSEEK_PROXY"}
    allowed.update({"TYPESAFE_API_KEY", "JEV_BASE_URL", "JEV_MODEL", "JEV_TIMEOUT_SECONDS", "JEV_PROXY"})
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#"):
            if line.lstrip("# ").startswith("model-name:"):
                result.setdefault("DEEPSEEK_MODEL", line.split(":", 1)[1].strip())
            continue
        line = line.removeprefix("export ")
        separator = "=" if "=" in line else ":"
        if separator not in line:
            continue
        name, value = (s.strip() for s in line.split(separator, 1))
        if name not in allowed:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            result[name] = value
    return result


@dataclass(frozen=True)
class ModelSettings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: float = 8.0
    thinking: str = "disabled"

    @classmethod
    def load(cls, root: Path, reference: Path | None = None) -> "ModelSettings":
        sources = [read_model_env(reference or root.parent / "PuwuQuiz" / ".env"),
                   read_model_env(root / ".env"), os.environ]
        values: dict[str, str] = {}
        key = ""
        for source in sources:
            values.update({k: v for k, v in source.items() if k.startswith("DEEPSEEK_") and v})
            key = source.get("DEEPSEEK_API_KEY") or source.get("API_KEY") or key
        result = cls(key, values.get("DEEPSEEK_BASE_URL", cls.base_url),
                     values.get("DEEPSEEK_MODEL", cls.model), float(values.get("DEEPSEEK_TIMEOUT_SECONDS", "8")),
                     values.get("DEEPSEEK_THINKING", "disabled"))
        url = urlsplit(result.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("API base must be an http(s) URL without credentials/query/fragment")
        if not 0 < result.timeout < float("inf"):
            raise ValueError("API timeout must be finite and positive")
        if result.thinking not in ("enabled", "disabled"):
            raise ValueError("DEEPSEEK_THINKING must be enabled or disabled")
        return result


class DeepSeekPolicy:
    def __init__(self, settings: ModelSettings, client: httpx.AsyncClient | None = None):
        if not settings.api_key:
            raise StrategyError("Missing DEEPSEEK_API_KEY/API_KEY in environment or local/reference env file")
        self.settings = settings
        # DeepSeek uses a direct connection even when the process has proxy vars.
        self.client = client or httpx.AsyncClient(
            timeout=settings.timeout, proxy=None, trust_env=False, follow_redirects=False)
        self.runner = asyncio.Runner()
        self.last_timing: dict = {}

    def close(self) -> None:
        try:
            self.runner.run(self.aclose())
        finally:
            self.runner.close()

    async def aclose(self) -> None:
        await self.client.aclose()

    def choose(self, state: State, actions: list[Action], *, deadline: float | None = None,
               stream: bool = True) -> Action:
        return self.runner.run(self.achoose(state, actions, deadline=deadline, stream=stream))

    def payload(self, state: State, actions: list[Action], stream: bool = True) -> dict:
        if not actions:
            raise StrategyError("No legal candidates")
        return {
            "model": self.settings.model,
            "stream": stream,
            "thinking": {"type": self.settings.thinking},
            "messages": [
                {"role": "system", "content": (model_instructions(state) + '\n' +
                    'Return only a JSON object {"action_id":"exact candidate ID"}.')},
                {"role": "user", "content": json.dumps({"state": state.for_model(),
                    "actions": [a.for_model() for a in actions]}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 128,
        }

    @staticmethod
    def _select(content: str, actions: list[Action]) -> Action:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        try:
            answer = json.loads(content, object_pairs_hook=unique_object)
            if not isinstance(answer, dict) or set(answer) != {"action_id"} or not isinstance(answer["action_id"], str):
                raise ValueError("Unexpected schema")
            return next(a for a in actions if a.id == answer["action_id"])
        except (ValueError, KeyError, TypeError, IndexError, StopIteration):
            raise StrategyError("Model response did not select an exact legal candidate") from None

    async def achoose(self, state: State, actions: list[Action], *, deadline: float | None = None,
                      stream: bool = True) -> Action:
        started = time.monotonic()
        timing = self.last_timing = {
            "route": "direct",
            "stream": stream, "first_event_seconds": None, "first_content_seconds": None,
            "first_valid_json_seconds": None, "decision_seconds": None, "total_seconds": None,
            "finish_reason": None, "status": "pending", "content_chunks": 0,
        }
        budget = self.settings.timeout
        if deadline is not None:
            budget = min(budget, deadline - started)
        try:
            if budget <= 0:
                raise TimeoutError
            payload = self.payload(state, actions, stream)
            timing["request_bytes"] = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            # A wall-clock deadline, not just a per-chunk idle timeout: trickling
            # keepalive/comments cannot prolong a decision indefinitely.
            async with asyncio.timeout(budget):
                async with self.client.stream(
                    "POST", self.settings.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.api_key}"},
                    json=payload, timeout=budget,
                ) as response:
                    if response.status_code != 200:
                        raise StrategyError(f"Model returned HTTP {response.status_code}; no action executed")
                    if stream:
                        selected = await self._read_sse(response, actions, started)
                    else:
                        await response.aread()
                        body = response.json()
                        choice = body["choices"][0]
                        timing["finish_reason"] = choice["finish_reason"]
                        if choice["finish_reason"] != "stop":
                            raise StrategyError("Model response did not finish normally")
                        selected = self._select(choice["message"]["content"], actions)
                        timing["usage"] = body.get("usage")
                    timing["decision_seconds"] = time.monotonic() - started
                    timing["status"] = "ok"
                    return selected
        except (TimeoutError, httpx.TimeoutException):
            timing["status"] = "timeout"
            raise StrategyError("Model exceeded decision deadline; no action executed") from None
        except httpx.HTTPError as exc:
            timing["status"] = "transport_error"
            timing["error_type"] = type(exc).__name__
            raise StrategyError("Model transport error; no action executed") from None
        except (ValueError, KeyError, TypeError, IndexError):
            timing["status"] = "invalid_response"
            raise StrategyError("Malformed model response; no action executed") from None
        except StrategyError:
            timing["status"] = "invalid_response"
            raise
        except asyncio.CancelledError:
            timing["status"] = "cancelled"
            raise
        finally:
            timing["total_seconds"] = time.monotonic() - started

    async def _read_sse(self, response: httpx.Response, actions: list[Action], started: float) -> Action:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raise StrategyError("Expected an SSE streaming response")
        timing = self.last_timing
        content = ""
        event_lines = []
        selected = None
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                event_lines.append(line[5:].lstrip(" "))
                continue
            if line:  # Ignore SSE comments, id and event fields.
                continue
            if not event_lines:
                continue
            data, event_lines = "\n".join(event_lines), []
            elapsed = time.monotonic() - started
            if timing["first_event_seconds"] is None:
                timing["first_event_seconds"] = elapsed
            if data == "[DONE]":
                if selected is not None:
                    return selected
                break
            event = json.loads(data)
            if "error" in event:
                raise StrategyError("Provider returned an error inside the stream")
            if event.get("usage") is not None:
                timing["usage"] = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise StrategyError("Unexpected additional completion")
                delta = choice.get("delta", {})
                fragment = delta.get("content")
                if fragment:
                    if selected is not None:
                        raise StrategyError("Content received after completion marker")
                    if not isinstance(fragment, str):
                        raise StrategyError("Non-text model content")
                    timing["content_chunks"] += 1
                    if timing["first_content_seconds"] is None:
                        timing["first_content_seconds"] = elapsed
                    content += fragment
                    try:
                        self._select(content, actions)
                    except StrategyError:
                        pass
                    else:
                        if timing["first_valid_json_seconds"] is None:
                            timing["first_valid_json_seconds"] = elapsed
                reason = choice.get("finish_reason")
                if reason is not None:
                    timing["finish_reason"] = reason
                    if reason != "stop":
                        raise StrategyError("Model stream truncated or refused")
                    # A parseable prefix alone is not a completed answer. Wait for
                    # the provider's normal finish marker, then validate everything.
                    selected = self._select(content, actions)
                    # Consume [DONE] as well so the HTTP connection can be reused.
        raise StrategyError("Stream ended without a normal completion marker")


class LocalFallbackPolicy:
    """Model first; when it fails or runs out of time, the local efficiency rung
    answers from the same frozen legal actions, so the Agent commits its own
    action instead of waiting for a client takeover.

    The model gets the decision deadline minus ``reserve_seconds``; the reserve
    covers the caller's polling, pre-click recheck and input.
    """
    def __init__(self, model, *, reserve_seconds, local=None):
        if not math.isfinite(reserve_seconds) or reserve_seconds <= 0:
            raise ValueError("Local fallback reserve must be finite and positive")
        self.model = model
        self.local = local or EfficiencyPolicy()
        self.reserve_seconds = reserve_seconds
        self.last_timing = {}
        self.last_result = None

    def choose(self, state, actions, *, deadline=None):
        return asyncio.run(self.achoose(state, actions, deadline=deadline))

    async def achoose(self, state, actions, *, deadline=None):
        self.last_result = None
        model_deadline = None if deadline is None else deadline-self.reserve_seconds
        try:
            if model_deadline is not None and time.monotonic() >= model_deadline:
                raise StrategyError("No model budget before the local fallback reserve")
            if hasattr(self.model, "achoose"):
                action = await self.model.achoose(state, actions, deadline=model_deadline)
            else:
                action = self.model.choose(state, actions, deadline=model_deadline)
            if action not in actions:
                raise StrategyError("Model returned an action outside the legal set")
            self.last_timing = {**(getattr(self.model, "last_timing", None) or {}), "rung": "model"}
            self.last_result = getattr(self.model, "last_result", None)
            return action
        except StrategyError as error:
            action = self.local.choose(state, actions)
            self.last_timing = {"rung": "local", "model_error": str(error),
                                "model_timing": getattr(self.model, "last_timing", None)}
            self.last_result = {"backend": "local_efficiency", "action_id": action.id}
            return action

    async def aclose(self):
        if hasattr(self.model, "aclose"):
            await self.model.aclose()
            runner = getattr(self.model, "runner", None)
            if runner is not None:
                runner.close()
        elif hasattr(self.model, "close"):
            self.model.close()

    def close(self):
        if hasattr(self.model, "close"):
            self.model.close()


class EfficiencyPolicy:
    """Offline smoke-test baseline, not a calibrated competitive policy."""
    def choose(self, state: State, actions: list[Action], *, deadline: float | None = None) -> Action:
        if deadline is not None and time.monotonic() >= deadline:
            raise StrategyError("Decision deadline expired")
        for kind in ("tsumo", "ron", "nuki"):
            # A win, then kita (a free bonus tile in sanma) before any discard.
            if found := next((a for a in actions if a.kind == kind), None):
                return found
        discards = [a for a in actions if a.kind == "discard"]
        if discards:
            def number(value, missing):
                return value if type(value) in (int, float) else missing
            # Unknown features (e.g. dora unavailable) rank last, never crash.
            return min(discards, key=lambda a: (
                number(a.features.get("shanten"), 99), -number(a.features.get("ukeire"), 0),
                -number(a.features.get("retained_dora"), 0), a.id))
        if found := next((a for a in actions if a.kind == "pass"), None):
            return found
        raise StrategyError("Offline policy has no implemented choice for this decision")
