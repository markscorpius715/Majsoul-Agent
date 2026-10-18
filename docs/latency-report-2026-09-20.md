# deepseek-flash Mahjong decision timing

Date: 2026-09-20. The user's latest confirmation is that the friendly match is **5+20**. This run actually sent 96 requests, in two batches of 48 each; no game was run and no mouse clicks were sent.

## Final implemented retest results

24 requests per method, 8 situation types × 3 rounds. The latency percentiles in the table below are only for successful requests; the success rate denominator includes all failures.

| Metric | SSE streaming | Non-streaming control |
| --- | ---: | ---: |
| Success / attempts | 22 / 24 | 19 / 24 |
| Connection failures | 2 | 5 |
| Median first valid content | 0.628 s | N/A |
| P95 first valid content | 0.849 s | N/A |
| Median complete request | 0.721 s | 0.868 s |
| Mean complete request | 0.747 s | 0.853 s |
| P95 complete request | 0.984 s | 1.187 s |
| Slowest successful request | 1.017 s | 1.187 s |
| Feature computation + complete request P95 | 0.992 s | 1.203 s |
| Legal decision within 5 s / all attempts | 22 / 24 | 19 / 24 |

The first content segment is not a completed decision: the JSON may span multiple segments. Actual execution waits for complete content, `finish_reason=stop` and SSE `[DONE]`, then verifies the candidate ID and re-observes the game state. The interval from the first streaming segment to the complete result is short, so do not expect streaming to greatly reduce total decision latency.

## Initial-round results are also retained

The initial implementation closed the response early after receiving the normal completion marker, without yet consuming `[DONE]`. To end the stream and reclaim the connection more properly, it was changed to read the stream termination marker before retesting.

| Initial-round metric | SSE streaming | Non-streaming |
| --- | ---: | ---: |
| Success / attempts | 19 / 24 | 22 / 24 |
| Median first content | 0.677 s | N/A |
| Median complete request | 0.785 s | 0.841 s |
| P95 complete request | 1.260 s | 1.087 s |
| Connection failures | 5 | 2 |

Both batches had 7 `httpx.ConnectError` occurrences, totaling 14/96. **You cannot assert based on these two batches of samples that the connection-reclamation change eliminated the network problem, nor can you attribute the difference in failure rate between streaming and non-streaming to the model.** No automatic retry was performed, so failures were not masked by subsequent successful requests. There were no timeouts of connected requests or out-of-candidate outputs; this is only the result of this sample and is not a reliability guarantee.

## Test Method

- Uses the `deepseek-flash` service configured in the local `../PuwuQuiz/.env`; keys are not logged.
- Explicitly sets `thinking.type=disabled`, requires returning only `{"action_id":"valid candidate ID"}`, with an output cap of 128 tokens; the total request deadline during testing is 8 seconds.
- 8 categories of simulated situations: early-round scattered hand, tenpai efficiency, opponent riichi, seven pairs, yaku-bearing meld, ron response, nuki/North extraction choice, and discard-after-draw after riichi.
- Sends the complete visible state and the candidate features computed by the rules layer, with 1–15 candidates; request JSON is about 2.0–14.5 KB.
- Each round uses a different visible discard pile/dora configuration with a fixed seed; tile types and visible counts are verified by the rules layer. They are synthetic decision snapshots, not hand-by-hand replays of complete legal hands.
- Streaming and non-streaming requests for the same situation alternate in order, sharing a persistent HTTP client. Server-side caching and connection warm/cold state may affect latency.
- Model output IDs are validated against legal candidates, but this round **did not evaluate play strength or decision optimality**.
- Quantiles: median uses the conventional definition; P95 uses nearest-rank. The sample size is small and cannot be treated as a production latency upper bound.
- Timing includes HTTP request, SSE decoding, and action parsing; local feature computation is reported separately. **Does not include real screenshots, YOLO, screen-stability waiting, mouse clicks, game animations, or network match confirmation.**

## Implementation under 5+20

The live loop starts counting the base 5-second budget from the first observation to the actionable phase; the model deadline is that budget minus `action_margin_seconds` (default 1 second), and already-consumed time is deducted; re-observation and clicking must still check whether the base time has expired. The 20 seconds of extra time in the rules is not treated as a 25-second budget available every time.

This is a budget estimated from observation time. There is currently no countdown OCR, so when joining midway it is unknown how much time has already passed in this turn, and it cannot be guaranteed to equal the game's true remaining time. Only after live countdown recognition is completed can it be accurately decided whether and how to use the extra time.

Streaming reception uses asynchronous HTTP and a deadline for the entire request; continuous heartbeats also cannot extend the request indefinitely. Half-finished JSON, parseable prefixes before normal completion, truncated responses, and expired actions are all unclickable. Currently a model request failure stops the process and does not automatically switch to an unevaluated local policy. The screenshot main loop still resamples after one decision completes; what is implemented this round is SSE streaming reception of model responses, not a scheduler that fully parallelizes screenshots and inference.

## Reproduction and Raw Records

```powershell
.venv\Scripts\python -m majsoul benchmark --rounds 3 --modes stream buffered
.venv\Scripts\python -m pytest -q
```

When `--output` is not specified, a separate directory is generated each time. A fixed output directory uses exclusive creation to avoid overwriting old results.

- Initial-round summary: `artifacts/benchmark-stream-5plus20/summary.json`
- Final summary: `artifacts/benchmark-stream-drained-5plus20/summary.json`
- The `requests.jsonl` in both directories saves each situation, candidate IDs, actual choice, timing, success status, and usage; there is no authentication information.
- After this round's changes, 58 tests pass, covering chunked SSE, heartbeat total timeout, end markers, legal action validation, and base time budget.

Interface basis: [DeepSeek Chat Completions documentation](https://api-docs.deepseek.com/api/create-chat-completion/)、[thinking-mode switch](https://api-docs.deepseek.com/guides/thinking_mode/)。
