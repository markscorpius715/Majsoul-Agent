"""One cancellable policy request; game capture and input stay on the caller thread."""

import asyncio
from concurrent.futures import Future, InvalidStateError
from copy import deepcopy
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread

from ..strategy import StrategyError


@dataclass(frozen=True)
class ChoiceResult:
    action: object
    timing: dict | None
    backend: dict | None


class ChoiceFailure(StrategyError):
    def __init__(self, error, policy):
        super().__init__(str(error) if isinstance(error, StrategyError) else type(error).__name__)
        self.timing = deepcopy(getattr(policy, "last_timing", None))
        self.backend = deepcopy(getattr(policy, "last_result", None))


class DecisionWorker:
    """A bounded daemon worker owning policy I/O and cleanup on one event loop.

    Cancelling invalidates the public future immediately. The slot is not freed
    until the actual request has unwound, so an uncooperative adapter cannot
    accumulate threads or requests. Native async policies cancel their HTTP I/O.
    """

    def __init__(self, policy):
        self.policy = policy
        self._lock = Lock()
        self._queue = Queue(maxsize=1)
        self._job = None
        self._active = None
        self._thread = None
        self._closed = False

    def submit(self, state, actions, *, deadline):
        with self._lock:
            if self._closed or self._job is not None:
                return None
            future = Future()
            self._job = (future, deepcopy(state), deepcopy(actions), deadline)
            self._queue.put_nowait(self._job)
            if self._thread is None:
                self._thread = Thread(target=self._run, name="majsoul-decision", daemon=True)
                self._thread.start()
            return future

    def cancel(self, future):
        future.cancel()
        with self._lock:
            active = self._active
            if self._job is None or self._job[0] is not future:
                return
        if active is not None:
            loop, task = active
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The request's event loop has already finished.

    async def _choose(self, job):
        future, state, actions, deadline = job
        with self._lock:
            self._active = (asyncio.get_running_loop(), asyncio.current_task())
        if future.cancelled():
            raise asyncio.CancelledError
        if hasattr(self.policy, "achoose"):
            action = await self.policy.achoose(state, actions, deadline=deadline)
        else:
            action = self.policy.choose(state, actions, deadline=deadline)
        return ChoiceResult(action, deepcopy(getattr(self.policy, "last_timing", None)),
                            deepcopy(getattr(self.policy, "last_result", None)))

    def _run(self):
        with asyncio.Runner() as runner:
            try:
                while True:
                    job = self._queue.get()
                    if job is None:
                        break
                    future = job[0]
                    try:
                        result = runner.run(self._choose(job))
                        future.set_result(result)
                    except asyncio.CancelledError:
                        future.cancel()
                    except InvalidStateError:
                        pass  # Cancellation won the race with a late answer.
                    except Exception as exc:
                        try:
                            future.set_exception(ChoiceFailure(exc, self.policy))
                        except InvalidStateError:
                            pass
                    finally:
                        with self._lock:
                            self._active = None
                            self._job = None
                            closed = self._closed
                    if closed:
                        break
            finally:
                if hasattr(self.policy, "aclose"):
                    runner.run(self.policy.aclose())
                    policy_runner = getattr(self.policy, "runner", None)
                    if policy_runner is not None:
                        policy_runner.close()
                elif hasattr(self.policy, "close"):
                    self.policy.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._job[0] if self._job is not None else None
            unused = self._thread is None
            if not unused and future is None:
                self._queue.put_nowait(None)
        if future is not None:
            self.cancel(future)
        if unused and hasattr(self.policy, "close"):
            self.policy.close()
