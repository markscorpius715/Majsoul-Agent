# GOAL_PROMPT: The Agent plays whole matches by itself

Version: 2026-10-18 (second revision: takeover supervisor retired). Working directory: `D:\pScript\majsoul`.

This file stands on its own. Do **not** take direction from `GOAL_PROMPT.md` or from the earlier phase plans. Repository rules in `AGENTS.md` (language, secrets, network routing, ROI projection, sanma-only state, test command) still apply. Read `MEMORY.md` for evidence, not for direction. The user's later instructions override this file.

Use this file as the prompt for a work loop. Each iteration follows §6 and ends with a written status (§9).

---

## 1. Goal and architecture

The Agent plays a three-player East match **entirely by itself**: every discard, call, pass, riichi, nuki, and win, from the first deal through the final result and back to the confirmed lobby.

**Architecture (user decision, 2026-10-18).** The Agent owns all game input for the whole match. No other process plays for it, and client auto-play is never switched on. Its own fallback chain, in order:

1. **Model:** DeepSeek only (`decision.policy: deepseek`). Measured p50 0.98 s / p95 1.49 s.
2. **Local efficiency:** `EfficiencyPolicy` on the same frozen snapshot. It answers when the model misses `deadline − decision.local_fallback_seconds`.
3. **Tsumogiri/PASS:** `emergency`, the user's rule. When our own hand area is clearly recognized, discard the drawn tile, or click a recognized PASS. It fires `emergency.lead_seconds` before the estimated client deadline, or at `emergency.maximum_turn_seconds`.
4. **The client's own timeout,** which also discards the drawn tile and skips calls.

**Time control (`majsoul/live/turn_timer.py`):** 7 s base per turn, plus a 20 s reserve shared by the turns of a hand and restored each new hand. Once the reserve is spent, turns are 5 s. The clock is read once per turn and counted down locally; the on-screen reading is the authority. An ordinary turn spends at most `emergency.reserve_per_turn_seconds` (2 s) of reserve; a riichi tile selection may use the rest. The deadlines are strictly ordered: model < local < tsumogiri < client timeout.

**Decisions:**
- A legal ron/tsumo is always taken (`decision.always_win`).
- Riichi is one choice, `riichi:<index>:<tile>`: the button is clicked, then the planned tile, with no second model call.
- Our own riichi status comes from our own verified declaration, never from the stick detector.
- An expired selection or an unresolved kan is not fatal.

**Watchdog.** `majsoul/live/watchdog.py`, started by `run --execute`, never sends game input. It only keeps an Agent alive:
- a crashed, stalled or unstartable Agent is fenced;
- a pre-loaded standby is promoted, or a cold Agent joins the running match;
- restarts are budgeted and backed off;
- a closed window or the operator's STOP ends the run.

Reviewed in-game popups (away-from-keyboard, connection) are dismissed by the Agent itself (`AgentSession`, `modal_manifest`). **Play is continuous:** after a match returns to the lobby, a fresh Agent enters the next one, until STOP/Ctrl+C or `watchdog.max_matches`.

The retired takeover supervisor (`majsoul/live/protection.py`: `run_protected`, `ProtectionMonitor`, the protector and handback) is kept in the repository but is not part of this workflow. Do not route new work through it.

**Done when:** 3 consecutive live matches (in one continuous run or several) appear in `watchdog-result.json` `matches` with:
- `agent_acceptance: true` for each match: returned to the lobby with **zero** `agent_restarts`;
- local-efficiency and tsumogiri rungs used on ≤ 10% of own turns. These rungs are the Agent playing by itself, but frequent use means the model path is failing.

**Recovery is proven when:** in live runs, at least 3 Agent restarts (a standby promotion or a cold join) are each followed by verified Agent moves in the same or the next hand, and the match still returns to the lobby.

The following never count toward done: offline passes, dry runs, and replays.

## 2. Evidence so far (2026-10-18)

- **Supervised live runs before the redesign:** 13. 0 accepted, 2 model decisions in total. First takeovers were mostly the supervisor misreading a normal state (unknown clock ×3, startup observer and animation trips) or the Agent stalling at startup. Nothing failed because of mahjong strength.
- **Last run, `protected-20261018-151224`:** a reserve-only clock (`20`) merged with our North tile and was read as unknown. One sample during a 4.4 s observer gap tripped the takeover, and the Agent was killed while about to click. Taking control back failed because our clipped North tile was counted as an unknown button on every frame.
- **Fixed offline (not yet verified live):**
  - one-read clock;
  - North-tile edge clutter in the button reader (2,297 phantom candidates in 5,371 frames, 0 real buttons affected);
  - drawer wait;
  - DeepSeek-only with the local rung;
  - re-timed tsumogiri rung;
  - Agent popup handling;
  - the watchdog;
  - the triage tool;
  - the takeover-frame replay.
- **Baseline ledger:** `artifacts/live/workflow-audit/stability/ledger.csv`.

## 3. Strategy

1. **Every lost turn is a defect.** Classify each one:
   - **(a) perception or timing:** the Agent misread a normal state, fell to a lower rung, or did not act;
   - **(b) process:** the Agent crashed, stalled or could not start, so the watchdog restarted it;
   - **(c) external:** lock screen, closed window, network, a popup that needs the user.

   Fix (a) and (b) so the class cannot recur. A standby promotion limits the damage, but the restart cause still has to be fixed.
2. **Fallback rungs are measured, not hidden.** Each local or tsumogiri decision is logged (`model_timing.rung`, `emergency_clicked`). Review why the model path missed.
3. **Unknowns are handled in the Agent.** A transient unknown (animation, overlay, clipped glyph) leads to a re-read using the last confirmed facts within their proven validity. It is never a reason to die or stall.
4. **Iterate on cheap evidence.** Use corpus replay of all saved frames, `tools/rejoin_replay.py` on recorded sequences, and live runs. Shadow matches (client auto-play with an observe-only Agent) are optional.
5. **Performance budget.** Fewer processes now run: no observer and no protector. The standby holds a second model copy; measure it on `.venv-gpu`.

**Out of scope:** mahjong strength beyond legality and basic efficiency, new features, other devices or layouts, four-player play, and collection-pipeline features.

## 4. Safety invariants

- **No helper plays the game.** The watchdog sends no game input. Its only input-like action is releasing a held left mouse button after fencing an Agent that died mid-click. It never enables client auto-play and never starts a match.
- **One input owner at a time.** A standby captures nothing and is refused input until promoted, after the old Agent has been fenced. dry-run sends no input. Matches run back to back only through a fresh Agent entering from the lobby; STOP/Ctrl+C ends the run. Never operate the lock screen, login, or client exit.
- **Fallback timing.** The tsumogiri rung only acts on our own hand area when it is clearly recognized: a valid hand, a separated drawn tile, recognized buttons, and the riichi restriction respected. It never clicks after `last_safe_input`. Timing changes need frame evidence, plus user approval when they relax an agreed rule (for example, the 30 s startup recognition).
- **Never fake knowledge.** Unknown stays unknown. The model chooses only from rule-layer legal actions.
- **Live runs need the user's authorization.** Otherwise, work offline and hand the user the exact command.
- **Fix the general cause.** No code for a specific screenshot, hand or test. Thresholds and geometry stay in config. Do not weaken assertions to make tests pass.

## 5. Stages and gates

| Stage | What | Gate |
|---|---|---|
| **S0: Instruments** | `tools/run_triage.py` (both `protected-*` and `live-*` runs), the ledger, `tools/rejoin_replay.py` | Done 2026-10-18 |
| **S1: Corpus** | Replay all saved frames through the clock, button, hand and lifecycle readers | 0 unknown clocks on frames a human labels readable; 0 phantom button candidates |
| **S2: Live hand** | The Agent plays live under the watchdog | First hand finished by the Agent with 0 restarts |
| **S3: Live match** | The Agent plays live | 3 consecutive accepted matches (§1) |

## 6. Loop protocol (one iteration)

1. **Orient.** Read the ledger and the latest `MEMORY.md` entry. Name the stage and the highest-priority open defect.
2. **Get evidence.** Use a new live run from the user, or the offline corpus. Run `python -m tools.run_triage <run>`.
3. **Classify the first lost turn or restart** as (a), (b) or (c). Open the frame; never diagnose from log text alone.
4. **Reproduce.** Add a fixture with provenance and a failing test.
5. **Fix the class.** Run targeted tests and the full suite: `.venv\Scripts\python -m pytest --basetemp <writable dir>`. Tests that need rapidocr run under `.venv-gpu`. New failures are not allowed; known pre-existing failures must stay unchanged and be listed.
6. **Sweep.** Re-run the corpus replay and report every reading that changed.
7. **Record.** Add a ledger row and a dated `MEMORY.md` entry (cause, evidence, fix, tests, stage, live status).
8. **Next evidence.**

**Stop and escalate when:**
- the same root cause recurs after its fix;
- a fix needs an invariant change;
- an external blocker persists;
- three iterations pass without progress.

## 7. Work items

**Done (2026-10-18, offline, after `live-20261018-164552-978100`):**
- **Own riichi:** fixes the tsumogiri lock that discarded the third haku.
- **Ron button re-read:** the 和 glyph reaches above the row; this fixes the two declined rons.
- **Fallback rules:** wins first; never PASS on our own turn; separate call and draw turns; deadline reset after a fallback click.
- **Compound riichi,** with no fatal expiry.
- **7+20 hand timer.**
- **Continuous play.**

**Done (2026-10-18, offline, earlier):**
- **T0:** triage tool and ledger.
- **T1:** one-read clock.
- **T3:** DeepSeek-only, local rung, re-timed tsumogiri rung.
- **T4 (buttons):** North-tile edge clutter.
- **T8:** clean termination: `window_gone`, restart budget, result always written.
- **Watchdog:** replaces takeover and handback.
- **Agent popup handling.**

**Open, in priority order:**
- **T4 (clock):** a reserve-only `20` merging with the North tile still misreads (`protected-20261018-151224/observer/000016.jpg`; 2026-10-11 frames 1351/001343). This matters when a turn's first reading comes after the base time ran out, because then only base time is assumed. Exclude the North-tile geometry from the clock masks; do not just widen HSV thresholds.
- **T4 (click target):** the 2026-10-11 PASS click overlapped a displayed North tile. Check click targets against the North-tile geometry.
- **T6 (performance):** measure per-process loop time with a standby loaded; cap torch threads; confirm CUDA. Earlier evidence: 4.4 s observer gaps and faulthandler dumps inside CNN `classify_many`.
- **T7 (startup):** recognition used 24.8 of 30 s in 151224; separate loading and dealing time from recognition failures. After 30 s the Agent now restarts and joins instead of handing over.
- **T10 (rung review):** after each live run, review every local and tsumogiri decision and why the model path missed.
- **T2 (optional):** shadow-match mode.

## 8. Metrics (in `triage.json` and the ledger)

`agent_acceptance`, `returned_lobby`, `agent_restarts`, `standby_promotions`, first restart reason, `decisions`, `verified/clicked`, decision latency p50/p95, `local_rung_decisions`, `emergency_rung_clicks`, `modal_clicks`, `base_expired_turns`, `startup_recognition_s`, `end_reason`.

**Progress means:** fewer restarts, fewer lower-rung decisions, and more hands finished by the Agent.

## 9. Status format at the end of each iteration

Write 4–8 lines:
- the stage and whether its gate is met;
- the evidence used;
- the first failure, its class and the frame;
- the fix and its test counts;
- the sweep delta;
- the next step and anything waiting on the user.

State plainly what was not verified live.
