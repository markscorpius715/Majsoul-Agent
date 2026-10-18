"""TypeSafe System One Choice adapter; no screen coordinates or mouse access."""
import asyncio
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from .domain import Action, State
from .strategy import StrategyError, read_model_env


@dataclass(frozen=True)
class JevSettings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"
    timeout: float = 8.0
    proxy: str | None = field(default=None, repr=False)

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Jev base URL must be http(s), without credentials/query/fragment")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Jev timeout must be finite and positive")
        if not self.model.strip():
            raise ValueError("Jev model must not be empty")

    @classmethod
    def load(cls, root: Path):
        values = read_model_env(root / ".env")
        values.update({k: v for k, v in os.environ.items() if v and (k.startswith("JEV_") or k == "TYPESAFE_API_KEY")})
        return cls(values.get("TYPESAFE_API_KEY", ""), values.get("JEV_BASE_URL", cls.base_url),
                   values.get("JEV_MODEL", cls.model), float(values.get("JEV_TIMEOUT_SECONDS", "8")),
                   values.get("JEV_PROXY") or None)


class JevPolicy:
    def __init__(self, settings: JevSettings, client: httpx.AsyncClient | None = None):
        if not settings.api_key:
            raise StrategyError("Missing TYPESAFE_API_KEY in environment or project .env")
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=settings.timeout, proxy=settings.proxy, trust_env=False, follow_redirects=False)
        self.runner = asyncio.Runner()
        self.last_timing = {}
        self.last_result = None

    def close(self):
        try:
            self.runner.run(self.aclose())
        finally:
            self.runner.close()

    async def aclose(self):
        await self.client.aclose()

    def choose(self, state: State, actions: list[Action], *, deadline: float | None = None) -> Action:
        return self.runner.run(self.achoose(state, actions, deadline=deadline))

    def payload(self, state, actions):
        from .policy_prompt import model_instructions
        ids = [a.id for a in actions]
        if not 1 <= len(ids) <= 255 or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise StrategyError("Jev requires 1..255 uniquely named legal candidates")
        return {"model": self.settings.model, "state": state.for_model(), "questions": {
            "action": {"type": "choice", "instructions": model_instructions(state),
                "criteria": {a.id: a.for_model() for a in actions}}}}

    @staticmethod
    def problem(body, actions):
        """The first reason a Choice response is unusable, or None."""
        def probability(value):
            return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
        try:
            answer = body["answers"]["action"]
        except (KeyError, TypeError):
            return "missing answers.action"
        if not isinstance(answer, dict):
            return "answers.action is not an object"
        choices = {a.id for a in actions}
        selected, distribution = answer.get("choice"), answer.get("probabilities")
        valid = isinstance(distribution, dict) and all(probability(p) for p in distribution.values())
        checks = (
            (answer.get("type") != "choice", "type is not choice"),
            (not isinstance(selected, str) or selected not in choices, "choice is not a legal candidate"),
            (not probability(answer.get("confidence")), "confidence is not a probability"),
            (not isinstance(distribution, dict), "probabilities is not an object"),
            (isinstance(distribution, dict) and set(distribution) != choices,
             "probabilities do not cover exactly the candidates"),
            (isinstance(distribution, dict) and not valid, "a probability is out of range"),
            (valid and not math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6),
             "probabilities do not sum to 1"),
            (valid and selected in distribution and distribution[selected] != max(distribution.values()),
             "choice is not the most probable"),
            (not isinstance(body, dict) or "model" not in body, "missing model"),
        )
        return next((reason for failed, reason in checks if failed), None)

    @classmethod
    def parse(cls, body, actions):
        reason = cls.problem(body, actions)
        if reason is not None:
            raise StrategyError(f"Invalid Jev Choice response ({reason}); no action executed")
        answer = body["answers"]["action"]
        selected = answer["choice"]
        return {a.id: a for a in actions}[selected], {"backend": "jev", "action_id": selected,
            "model": body["model"], "confidence": answer["confidence"],
            "probabilities": answer["probabilities"]}

    async def achoose(self, state, actions, *, deadline=None):
        started = time.monotonic()
        self.last_result = None
        self.last_timing = {"backend": "jev", "status": "started",
                            "route": "proxy" if self.settings.proxy else "direct"}
        end = min(deadline, started + self.settings.timeout) if deadline is not None else started + self.settings.timeout
        try:
            payload = self.payload(state, actions)
            if time.monotonic() >= end:
                raise TimeoutError
            async with asyncio.timeout(end - time.monotonic()):
                response = await self.client.post(self.settings.base_url.rstrip("/") + "/v1/systemone",
                    headers={"Authorization": "Bearer " + self.settings.api_key}, json=payload)
                self.last_timing["http_status"] = response.status_code
                if response.status_code != 200:
                    self.last_timing["status"] = "http_error"
                    raise StrategyError(f"Jev HTTP {response.status_code}; no action executed")
                action, result = self.parse(response.json(), actions)
                if time.monotonic() >= end:
                    raise TimeoutError
                self.last_result = result
                self.last_timing["status"] = "ok"
                return action
        except (TimeoutError, httpx.TimeoutException):
            self.last_timing["status"] = "timeout"
            raise StrategyError("Jev exceeded decision deadline; no action executed") from None
        except httpx.HTTPError as exc:
            self.last_timing["status"] = "transport_error"
            self.last_timing["error_type"] = type(exc).__name__
            raise StrategyError("Jev transport error; no action executed") from None
        except asyncio.CancelledError:
            self.last_timing["status"] = "cancelled"
            raise
        except (ValueError, StrategyError) as error:
            if self.last_timing.get("status") != "http_error":
                self.last_timing["status"] = "invalid_response"
            # Logged without secrets: which check failed, never the key.
            self.last_timing["reason"] = (str(error) if isinstance(error, StrategyError)
                                          else "response is not JSON")
            if isinstance(error, StrategyError):
                raise
            raise StrategyError("Malformed Jev response; no action executed") from None
        finally:
            self.last_timing["total_seconds"] = time.monotonic() - started


class FallbackPolicy:
    """After primary failure, race a fresh Jev request against DeepSeek.

    Both recovery branches share the original deadline; only the first valid
    result is returned. Factories stay lazy until their backend is needed.
    """
    def __init__(self, primary_factory, fallback_factory, timeout=8.0, hedge_seconds=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Fallback timeout must be finite and positive")
        if hedge_seconds is not None and (not math.isfinite(hedge_seconds) or hedge_seconds <= 0):
            raise ValueError("Hedge delay must be finite and positive")
        self.factories = (primary_factory, fallback_factory)
        self.policies = {}
        self.timeout = timeout
        self.hedge_seconds = hedge_seconds
        self.last_timing = {}
        self.last_result = None
        self.runner = asyncio.Runner()

    def choose(self, state, actions, *, deadline=None):
        return self.runner.run(self.achoose(state, actions, deadline=deadline))

    async def achoose(self, state, actions, *, deadline=None):
        end = min(deadline, time.monotonic() + self.timeout) if deadline is not None else time.monotonic() + self.timeout
        self.last_result = None
        self.last_timing = {"attempts": []}
        tasks = []
        try:
            async with asyncio.timeout(max(0, end - time.monotonic())):
                try:
                    if self.hedge_seconds is None:
                        result = await self._attempt(0, 1, state, actions, end)
                    else:
                        result = await self._hedged(state, actions, end, tasks)
                except StrategyError:
                    if time.monotonic() >= end:
                        raise StrategyError("No decision budget remains for backend fallback") from None
                    self.last_timing["recovery"] = "parallel_jev_restart_and_deepseek"
                    tasks = [
                        asyncio.create_task(self._attempt(0, 2, state, actions, end, restart=True)),
                        asyncio.create_task(self._attempt(1, 1, state, actions, end)),
                    ]
                    pending = set(tasks)
                    result = None
                    while pending and result is None:
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for task in tasks:
                            if task not in done:
                                continue
                            try:
                                result = task.result()
                                break
                            except StrategyError:
                                continue
                    if result is None:
                        raise StrategyError("Jev retry and DeepSeek fallback both failed; no action executed")
                action, backend, attempt, model_result = result
                if time.monotonic() >= end:
                    raise StrategyError("Backend returned a late action; no action executed")
                self.last_result = model_result
                self.last_timing.update(selected_backend=backend, selected_attempt=attempt)
                return action
        except TimeoutError:
            raise StrategyError("Decision backends exceeded shared deadline; no action executed") from None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _hedged(self, state, actions, end, tasks):
        """Primary first; if it is still silent after ``hedge_seconds``, race
        the fallback too. A primary failure before then falls through to the
        existing parallel recovery."""
        primary = asyncio.create_task(self._attempt(0, 1, state, actions, end))
        tasks.append(primary)
        done, _ = await asyncio.wait({primary}, timeout=self.hedge_seconds)
        if done:
            return primary.result()
        self.last_timing["recovery"] = "hedged_deepseek"
        hedge = asyncio.create_task(self._attempt(1, 1, state, actions, end))
        tasks.append(hedge)
        pending = {primary, hedge}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in (primary, hedge):
                if task in done:
                    try:
                        return task.result()
                    except StrategyError:
                        continue
        raise StrategyError("Jev and the hedged DeepSeek request both failed; no action executed")

    async def _attempt(self, index, attempt, state, actions, end, *, restart=False):
        backend = "jev" if index == 0 else "deepseek"
        record = {"backend": backend, "attempt": attempt, "restarted": restart,
                  "status": "started"}
        self.last_timing["attempts"].append(record)
        started = time.monotonic()
        policy = None
        status = "unavailable"
        try:
            if restart:
                previous = self.policies.pop(index, None)
                if previous is not None:
                    await self._close_policy(previous)
            if time.monotonic() >= end:
                status = "timeout"
                raise StrategyError("No decision budget remains for backend request")
            if index not in self.policies:
                self.policies[index] = self.factories[index]()
            policy = self.policies[index]
            status = "failed"
            if hasattr(policy, "achoose"):
                action = await policy.achoose(state, actions, deadline=end)
            else:
                action = policy.choose(state, actions, deadline=end)
            if time.monotonic() >= end or action not in actions:
                status = "invalid_action"
                raise StrategyError("Backend returned a late or illegal action")
            status = "ok"
            return action, backend, attempt, getattr(policy, "last_result", None)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            record.update(getattr(policy, "last_timing", {}))
            if status in ("cancelled", "invalid_action") or record["status"] == "started":
                record["status"] = status
            record["total_seconds"] = time.monotonic() - started

    @staticmethod
    async def _close_policy(policy):
        try:
            if hasattr(policy, "aclose"):
                await policy.aclose()
            elif hasattr(policy, "close"):
                policy.close()
        finally:
            runner = getattr(policy, "runner", None)
            if runner is not None:
                runner.close()

    def close(self):
        try:
            self.runner.run(self.aclose())
        finally:
            self.runner.close()

    async def aclose(self):
        for policy in self.policies.values():
            await self._close_policy(policy)
