# GOAL_PROMPT.md

## 1. Goal

Implement a Mahjong Soul mahjong Agent workflow based on pure visual perception.

The system continuously acquires frames from the Mahjong Soul game window, restores the complete mahjong state through fixed ROIs (Region of Interest) and lightweight visual methods, constructs a structured `MahjongState` and a finite set of legal actions `LegalActions` when a decision is needed, calls the decision model to choose an action, then executes mouse operations, and verifies whether the action succeeded through subsequent frames.

Current stage:

- Do not use YOLO as the primary visual solution.
- Use fixed ROI + image preprocessing + OCR / Template Matching / Tile Classifier + temporal state tracking.
- Jev is the target decision backend.
- The Jev official Choice interface has been implemented and real-call verification has been completed; before the fixed ROI state layer and interaction trigger layer pass acceptance, it will not be connected to the real-time clicking chain.
- `deepseek-flash` remains as a fallback under the unified decision interface.
- The decision backend must be abstracted into a unified interface, so that future integration of Jev requires no modification to modules such as perception, state tracker, and executor.
- It is not allowed to hardcode special rules for a particular match recording, a particular hand, or a particular prompt in order to pass tests.
- The system design should support different match states and continuous operation throughout a complete match.

Final workflow:

Game Window
    ↓
Frame Capture
    ↓
ROI Extraction
    ↓
Visual Perception
    ↓
State Tracker
    ↓
MahjongState
    ↓
Trigger Detection
    ↓
Legal Action Generation
    ↓
Decision Backend
    ├── Jev            [target]
    └── DeepSeek-Flash [current fallback]
    ↓
Action
    ↓
Mouse Executor
    ↓
Visual Verification
    ↓
State Update
    └──────────── loop


# 2. Core Principle

The system must clearly distinguish the following four layers:

1. Observation
Visual information that can be directly observed in the current frame.

2. State
The complete mahjong state recovered from the current frame + historical state.

3. Decision
Select an action based on the complete state and the set of legal actions.

4. Execution
Convert the abstract action into actual mouse operations, and confirm whether it succeeded.

It is forbidden to mix the above four layers together.

For example, one should not:

Screenshot
→ LLM
→ click

Instead, one should:

Screenshot
→ structured observation
→ MahjongState
→ legal actions
→ decision
→ verified action


# 3. Scope

Phase one prioritizes support for:

- The Mahjong Soul PC / Web UI used in the current video.
- Fixed window ratio and layout.
- The match mode actually used by the current project.
- At the same time, try to avoid hard-coding the code to a fixed number of players.

Because Mahjong Soul may have:

- Three-player mahjong
- Four-player mahjong

Therefore:

`player_count`
`seat_order`
`river_count`
`meld_count`

etc. should all be obtained from configuration or state, rather than being `3` / `4` constants scattered throughout the code.

In particular, do not hard-code the entire system as sanma just because the current test video shows "North extraction".


# 4. Directory Structure

It is recommended to ultimately form:

src/
├── capture/
│   ├── screen_capture.py
│   └── frame_source.py
│
├── calibration/
│   ├── roi_editor.py
│   ├── roi_config.py
│   └── roi_validator.py
│
├── perception/
│   ├── roi_extractor.py
│   ├── tile_classifier.py
│   ├── tile_segmenter.py
│   ├── river_reader.py
│   ├── hand_reader.py
│   ├── meld_reader.py
│   ├── score_reader.py
│   ├── round_reader.py
│   ├── action_reader.py
│   └── template_matcher.py
│
├── state/
│   ├── schemas.py
│   ├── state_tracker.py
│   ├── event_detector.py
│   └── state_validator.py
│
├── rules/
│   ├── legal_actions.py
│   ├── tile_utils.py
│   └── mahjong_rules.py
│
├── decision/
│   ├── base.py
│   ├── deepseek_flash.py
│   ├── jev.py
│   ├── prompt_builder.py
│   └── response_parser.py
│
├── executor/
│   ├── action_mapper.py
│   ├── mouse_executor.py
│   └── action_verifier.py
│
├── runtime/
│   ├── agent_loop.py
│   └── controller.py
│
├── logging/
│   ├── event_logger.py
│   └── replay_recorder.py
│
└── tests/

configs/
├── roi.yaml
├── vision.yaml
├── runtime.yaml
└── decision.yaml

artifacts/
├── classifiers/
├── templates/
├── debug_frames/
└── logs/


# 5. ROI Calibration

## 5.1 Do not manually annotate every frame

ROIs only need to be manually defined once on a reference screenshot.

Implementation:

`calibration/roi_editor.py`

Use:

`cv2.selectROI()`

Allow the user to sequentially box-select fixed UI regions on a reference frame.

Save normalized coordinates:

x1_norm = x1 / width
y1_norm = y1 / height
x2_norm = x2 / width
y2_norm = y2 / height

It is forbidden to save only absolute pixel coordinates.

When the target frame and the reference frame have different resolutions, first calculate the aspect ratio drift. When the drift is within the configurable automatic threshold,
scale separately according to the normalized `x/y` proportions; pixel dimensions or aspect ratios are not required to be exactly the same. When the threshold is exceeded, do not silently
apply a single mapping; you must output candidate crops including at least independent stretching, width-preserving multi-alignment, and height-preserving multi-alignment, and only after
an recognition completeness test or manual inspection can the projection be fixed. When the window theme, UI rearrangement, or candidates cannot be uniquely determined, recalibration
is still required.


## 5.2 Required ROI

At least consider:

### Global / central information

- `round_info`
  - East round / South round, etc.

- `remaining_tiles`
  - Remaining tile count

- `honba`
  - Honba count

- `riichi_sticks`
  - Number of riichi sticks in the pot on the table

- Each player's `score`

- seat / dealer indicators


### Dora

- `dora_indicators`

It is recommended to handle it as up to multiple slots, rather than only recognizing the one currently lit up.


### Hand

- `self_hand`

Need to distinguish:

- Original concealed hand
- Newly drawn tile

If there is an obvious gap between the newly drawn tile and the original hand in the UI, it should be explicitly recognized.


### Rivers

Dynamically maintain according to the number of players:

- `river_self`
- `river_player_1`
- `river_player_2`
- Optional `river_player_3`

Do not only read other players' discards.

One's own discard pile is likewise used for:

- Track tsumogiri/tegiri
- Riichi
- Furiten-related information
- Full state recovery


### Melds

Every player requires:

`melds[player]`

and not just oneself.

Identify:

- chi
- pon
- kan
- closed/open kan, if visually distinguishable
- kita / North extraction, if the corresponding mode exists


### Action Area

Must separately have:

`action_buttons`

Used to determine:

- Whether a decision is needed
- Which UI operations are currently allowed


### Turn / interaction indicators

Used to determine:

- Whether it is currently one's turn to discard
- Whether currently waiting for other players
- Whether a response window has appeared


# 6. Visual Perception Strategy

Principles:

Try not to use a single "large model" to solve all vision problems.

Use the simplest, most stable method for different ROIs.


## 6.1 OCR-like information

Used for:

- score
- remaining tiles
- round number
- honba
- riichi sticks

Priority order:

1. fixed-font template matching
2. digit classifier
3. OCR

Mahjong Soul fonts and positions are fixed, so do not rely on general OCR by default.


## 6.2 Tile Recognition

Need to recognize standard Japanese mahjong tiles:

- 1m ~ 9m
- 1p ~ 9p
- 1s ~ 9s
- east
- south
- west
- north
- white
- green
- red

as well as:

- aka 5m
- aka 5p
- aka 5s

Suggested internal unified representation:

1m ... 9m
1p ... 9p
1s ... 9s
1z ... 7z

0m
0p
0s

i.e. 37-class tile representation.


## 6.3 Hand

Do not directly perform object detection on the entire hand ROI.

Recommended:

self_hand ROI
    ↓
edge / spacing / geometry
    ↓
individual tile crops
    ↓
37-class classifier


## 6.4 Rivers

Since the discard ponds of each direction have different orientations, first perform orientation normalization.

For example:

left river
    ↓
rotate / rectify
    ↓
canonical orientation

top river
    ↓
rotate / rectify
    ↓
canonical orientation

right river
    ↓
rotate / rectify
    ↓
canonical orientation

Then unify:

river ROI
    ↓
grid / connected-component segmentation
    ↓
tile crops
    ↓
same tile classifier

The goal is:

All tiles ultimately enter the same classifier in approximately the same orientation.

Do not let the classifier meaninglessly learn:

- 0°
- 90°
- 180°
- 270°

four sets of visual distributions.


## 6.5 Riichi tile

Tiles placed horizontally in the discard pond carry additional semantics:

`riichi declaration`

Therefore the visual result should at least preserve:

{
    "tile": "5p",
    "orientation": "horizontal"
}

It must not simply rotate and lose the original orientation information.


# 7. Observation Schema

Each frame produces:

```python
FrameObservation(
    timestamp=...,

    round_info=...,
    remaining_tiles=...,
    honba=...,
    riichi_sticks=...,

    player_scores={...},

    hand=[...],
    drawn_tile=...,

    rivers={
        player_id: [...]
    },

    melds={
        player_id: [...]
    },

    kita={
        player_id: ...
    },

    dora_indicators=[...],

    action_buttons=[...],

    turn_indicator=...,

    confidence={
        ...
    }
)

Observation must allow:

unknown

It is forbidden to forcibly guess a result when perception is uncertain.

8. Mahjong State

Maintain a long-lived:

MahjongState

It is suggested to at least include:

MahjongState(
    mode=...,
    player_count=...,

    round=...,
    honba=...,
    riichi_sticks=...,
    remaining_tiles=...,

    dealer=...,
    self_seat=...,

    players={
        player_id: PlayerState(...)
    },

    hand=[...],
    drawn_tile=...,

    dora_indicators=[...],

    current_actor=...,

    last_discard=...,

    available_actions=[...],

    phase=...,

    history=[...]
)

Where:

PlayerState(
    seat=...,
    score=...,
    river=[...],
    melds=[...],
    riichi=False,
    kita_count=0,
    ...
)
9. State Tracker

Do not re-recover the entire match state every frame.

Adopt:

previous_state
+
current_observation
↓
event detection
↓
new_state

For example:

Old discards:

[3p, 7s, east]

New discards:

[3p, 7s, east, 5m]

Inference:

{
  "type": "DISCARD",
  "player": 1,
  "tile": "5m"
}

Phase 5 only restores `GameEvent`s that can be objectively proven by table-state diffs:

ROUND_START
DRAW
DISCARD
RIICHI
MELD
CHI
PON
KAN
KITA
SCORE_UPDATE
ROUND_END
RESYNC

`MELD` is a generic event used when chi, pon, and kan cannot yet be reliably distinguished; when they can be reliably distinguished, use
`CHI` / `PON` / `KAN`. Phase 5 does not answer why a player did not pon or why there was no ron,
nor does it infer `PASS` from ordinary table diffs.

Events and actions must be typed separately:

```text
GameEvent
├── DRAW / DISCARD / RIICHI
├── MELD / CHI / PON / KAN / KITA
├── RON / TSUMO
├── SCORE_UPDATE
├── ROUND_START / ROUND_END / GAME_END
└── RESYNC

AgentAction
├── DISCARD / RIICHI
├── CHI / PON / KAN / KITA
├── RON / TSUMO
└── PASS
```

`RON`, `TSUMO`, and `GAME_END` are GameEvents, but they are only produced after Phase 6's result prompts,
overlays, or interaction evidence. `PASS` belongs only to AgentAction, because it usually does not leave an independent event in the
table's visual state.
10. State Validation

Implement deterministic validation.

At minimum check:

Whether the number of concealed hand tiles is legal
The discards can only grow reasonably or change reasonably due to melds
Meld count changes are reasonable
The total occurrence count of the same specific tile cannot exceed four
The number of red tiles is reasonable
The current player state is legal
Round transition is reasonable
Score format is reasonable
Remaining tiles should not increase without reason

Conflict found:

Do not continue calling the decision model.

Execute:

RESYNC

That is:

Rescan multiple ROIs
+
Perform consistency judgment on the most recent frames.

11. Interaction / Decision Event Detection

Very important:

Do not use simple logic:

if my_turn:
    call_model()

Because after an opponent discards, you yourself may also need it:

ron
chi
pon
kan
pass

Correct logic:

if action_buttons_present:
    generate_legal_actions()
    call_decision()

elif self_discard_required:
    generate_legal_actions()
    call_decision()

else:
    update_state_only()

The decision model cannot be called continuously at 30 FPS.

Trigger only at discrete decision points.

The input for this phase is:

MahjongState
+
Action Area Observation
+
Result / Overlay Observation

New interaction events in this phase:

ACTION_WINDOW_OPEN
AVAILABLE_ACTIONS_CHANGED
DECISION_REQUIRED
ACTION_WINDOW_CLOSE

And when result UI evidence is available, fill in:

RON
TSUMO
GAME_END

AgentAction.PASS is recorded in the decision/execution log and is not disguised as a table GameEvent.

12. Legal Action Generation

Before entering the LLM / Jev, legal actions must be generated by the program.

For example:

[
  {
    "id": "discard_5p",
    "type": "discard",
    "tile": "5p"
  },
  {
    "id": "discard_7s",
    "type": "discard",
    "tile": "7s"
  }
]

Or:

[
  {"id": "pon_5p", "type": "pon"},
  {"id": "ron", "type": "ron"},
  {"id": "pass", "type": "pass"}
]

The model can only:

Choose from legal_actions.

The model is forbidden from inventing nonexistent actions.

13. Decision Backend Abstraction

A unified interface must be defined:

class DecisionBackend(ABC):

    @abstractmethod
    def decide(
        self,
        state: MahjongState,
        legal_actions: list[LegalAction],
    ) -> DecisionResult:
        ...

Return:

DecisionResult(
    action_id=...,
    confidence=...,
    raw_response=...,
    backend=...
)
14. DeepSeek-Flash Fallback

Current default:

decision:
  backend: deepseek_flash

Call the already-tested deepseek-flash API in the project.

Do not reimplement an incompatible API client; if a tested client already exists in the project, reuse it first.

14.1 Input

DeepSeek-Flash can only receive:

compressed structured state
legal actions
task instruction

Do not pass full screenshots.

Example:

{
  "state": {
    "round": "E2",
    "honba": 1,
    "remaining_tiles": 28,

    "self": {
      "seat": "S",
      "score": 26700,
      "hand": [
        "2m", "3m", "4m",
        "4p", "5p", "6p",
        "4s", "5s", "6s",
        "2s", "3s", "7s", "8s"
      ]
    },

    "dora_indicators": ["5s"],

    "players": {
      ...
    }
  },

  "legal_actions": [
    "discard_2s",
    "discard_3s",
    "discard_7s",
    "discard_8s"
  ]
}
14.2 Output

Require DeepSeek-Flash to strictly return:

{
  "action_id": "discard_7s",
  "reason": "...",
  "confidence": 0.73
}

Where:

action_id

Must belong to the input legal_actions.

Parsing failed:

retry once

Failed again:

Enter safe fallback.

15. Jev Backend

Reserved:

decision/jev.py

After obtaining the Jev API in the future, only replace:

DecisionBackend

Must not modify:

perception
state tracker
rules
executor

Jev should receive:

structured state
+
finite legal action space

And choose:

Choice(action)

If the Jev API can return:

candidate probability
confidence

Then save:

DecisionResult(
    action_id=...,
    confidence=...,
    probabilities={
        ...
    }
)

For subsequent analysis.

16. Backend Selection

Configuration:

decision:

  primary: jev
  fallback: deepseek_flash

  jev:
    enabled: false

  deepseek_flash:
    enabled: true

Current runtime logic:

Jev unavailable
↓
DeepSeek-Flash

Future:

Jev
↓ success
return

↓ API error
DeepSeek-Flash

Do not, because Jev is currently unavailable, have the following appear in other modules:

if using_deepseek:
   ...

Outside the decision module, the specific backend should not be known.

17. Decision Prompt

The DeepSeek-Flash prompt should be as short and structured as possible.

System description:

You are the decision module of a Riichi Mahjong agent.

You are given:

a structured game state
a set of legal actions

Select exactly one action from legal_actions.

Do not invent actions.
Do not modify the game state.
Return strict JSON only.

Optimize the Mahjong decision based on available information.

Output schema:

{
"action_id": "<one legal action id>",
"reason": "<short explanation>",
"confidence": <0 to 1>
}

Do not allow the LLM to:

Recognize tiles from screenshots
Determine button positions
Control the mouse
Judge on its own whether an action is legal
18. Action Executor

Abstract action:

discard_5p
pon
kan
riichi
ron
tsumo
pass
kita

By:

ActionMapper

Convert into the actual UI target.

For example discard:

hand tiles
+
recognized tile positions
↓
tile id → screen coordinate

Do not use a fixed:

click(942, 1017)

To represent "discard 5p".

Because the position of 5p in different hands changes.

Should use what perception returns:

DetectedTile(
    tile="5p",
    bbox=...
)
19. Execution Verification

After any action, must:

click
↓
wait for UI transition
↓
capture
↓
verify

For example:

discard 5p

Verify:

Hand decreases by one tile
Discard pile increases by 5p
turn changes

Cannot simply:

click
↓
state += discard

If validation fails:

retry / resync

Do not blindly continue to the next action.

20. Confidence and Uncertainty

The vision module returns confidence.

For example:

{
  "tile": "5p",
  "confidence": 0.96
}

When core information is below the threshold:

Do not call the decision backend.

For example:

unknown tile
score OCR conflict
river discontinuity
hand tile count invalid

Enter:

RESYNC

Unknown must be a first-class state and cannot be determined only by low classifier confidence. The Unknown Detector must at least combine:

1. model uncertainty: maximum probability and class margin insufficient;
2. state inconsistency: a fifth copy of the same tile, losing multiple hand tiles despite no meld, and other rule conflicts;
3. novel visual pattern: the visual embedding distance between the current crop and verified samples is too large.

On the first anomaly, first `RESYNC` and retry over the next 3–5 stable frames. Only if unknown persists, enter:

HUMAN_REVIEW
+
SAFE_PAUSE

Offline replay pauses processing; real-time operation only pauses the Agent's automatic actions and does not control or pause the Mahjong Soul client.
The same unknown visual state must be deduplicated and must not be notified repeatedly per video frame.

Each item pending review saves:

```text
unknown_cases/<case_id>/
├── full_frame.png
├── crop.png
├── previous_frame.png
├── next_frame.png
├── state.json
└── metadata.json
```

Feishu notifications only serve as a reminder that "there is a new case pending handling" and carry the case id, type, time, context image, and candidates.
The main labeling entry point is the local interactive GUI: it displays the crop, full frame, preceding and following frames, source, and candidates item by item, and the human clicks
the tile type or enters a label to explicitly confirm. The chat `/label <case_id> <label>` is retained only as a strictly parsed backup entry point,
and cannot replace the GUI's context review.

Explicit human confirmation in the GUI or backup command is written only to the verified dataset and training queue; it is forbidden to take a single new image
and immediately perform online SGD. The GUI must support pending/verified/rejected and task type filtering, skipping, rejection, and queue
progress, and reuse the same label/source validation and atomic manifest update logic as the CLI.

21. Frame Processing Strategy

Forbidden:

30 FPS × all OCR × all classifiers × LLM

Recommended:

capture 10~30 FPS

But perception is layered:

Cheap detector

Every frame:

frame diff
action area
turn indicator
Normal perception

Only when the state changes:

rivers
hand
melds
Expensive perception

Only:

round start
resync
OCR uncertainty

Re-read the complete state.

22. Runtime State Machine

Recommended:

WAIT_GAME
    ↓
ROUND_START
    ↓
WAIT_ACTION
    ↓
OBSERVE_EVENT
    ↓
STATE_UPDATE
    ↓
DECISION_REQUIRED?
    ├── NO → WAIT_ACTION
    │
    └── YES
          ↓
      GENERATE_LEGAL_ACTIONS
          ↓
      DECISION
          ↓
      EXECUTE
          ↓
      VERIFY
          ├── success → WAIT_ACTION
          └── failure → RESYNC

In addition:

ROUND_END
GAME_END
PAUSED
RESYNC

All should explicitly exist.

23. Logging

Each decision point saves:

timestamp
frame
cropped ROIs
MahjongState
legal_actions
backend
raw_model_response
selected_action
confidence
click_position
verification_result

For example:

artifacts/logs/
└── game_2026xxxx/
    ├── events.jsonl
    ├── decisions.jsonl
    └── frames/

This is extremely important for troubleshooting.

## 23.1 Continual Perception Learning Loop

Initial recognition data uses bootstrap rather than waiting for a recording to naturally cover all rare states:

```text
datasets/
├── raw/
│   ├── public/
│   ├── captured/
│   └── unknown/
├── verified/
│   ├── tiles/
│   ├── actions/
│   ├── melds/
│   └── ui_states/
├── rejected/
├── golden_set/
└── manifests/
    ├── dataset_v0.json
    └── ...
```

Majsoul-specific live data and manually verified data are used for target-domain training/validation with high weight; general mahjong
data can only serve as a low-weight warm start or pretraining source. Before any public data is introduced, its license,
class mapping, source, and training/validation leakage must be checked; it cannot be mixed into the dataset directly based only on a webpage's quantity description.

Recordings across play modes must record `player_count` and `mode` in the source/sample manifest. Four-player material can
supplement shared visual capabilities such as tile faces, glyphs, discards, melds, screen orientation, and general UI, but four-player score conservation, North extraction,
event rules, or mode-specific labels must not enter sanma rule/state training. Original videos are registered only as unlabeled sources; only single samples that have been cropped,
deduplicated, and manually confirmed may enter the verified dataset.

Each sample manifest records at least: id, source, label, verified_by, collection time, source frame,
model version, original prediction/confidence, dataset split, and content hash. Secrets and personal credentials must not enter the dataset.

Manually confirmed samples first enter the training queue. Training occurs only after the configured threshold is accumulated or the planned training window is reached.
candidate model. The candidate must pass regression on the versioned golden set and the historical validation set: old capabilities must not
degrade significantly, and new unknown cases must improve. Only after passing may it be promoted; both the old and new models and the data manifest must be
traceable; a failed candidate keeps its report but must not go live.

Complete closed loop:

```text
Bootstrap Data -> Dataset v0 -> Model v0 -> Perception
                                            ├── known -> continue
                                            └── persistent unknown
                                                  -> SAFE_PAUSE
                                                  -> review package
                                                  -> Feishu alert
                                                  -> local review GUI
                                                  -> human label
                                                  -> verified dataset
                                                  -> training queue
                                                  -> candidate model
                                                  -> regression gate
                                                  -> promote or reject
```

24. Replay Mode

Must support:

python -m app replay game.mp4

Replay mode:

Do not execute mouse operations
Run the full perception
state tracker
decision trigger
decision

Used for offline testing on the currently uploaded game recording.

25. Debug Visualization

Implement debug mode:

Draw on the original image:

ROI
tile label
tile confidence
score
river sequence
meld sequence
current actor
detected action buttons

For example:

[5p 0.99]
[6p 0.97]
[7p 0.96]

Output:

debug.mp4

Convenient for quickly observing at which layer the system makes an error.

26. Tests

At minimum establish the following tests.

ROI

Input a fixed reference frame.

Check:

ROI coordinates are valid.

Tile classifier

Test:

normal tile
red five
rotated river tile
small river tile
River parsing

Given a discard pond crop:

Output the correct order.

State transitions

For example:

discard event

pon event

riichi event

kita event

Legal action validation

Any action the model outputs that is not in:

legal_actions

the action in:

must be rejected.

Backend test

Mock：

Jev unavailable

Confirm:

DeepSeek-Flash automatically takes over.

Executor dry-run

Output:

target coordinate

But do not actually click.

27. Development Order

Do not complete all modules at once.

Strictly follow the order below.

Phase 1: Offline ROI

Complete:

extract reference frame
interactive ROI editor
roi.yaml
ROI debug visualization
Phase 2: Tile recognition

Complete:

self hand
one river
tile classifier

Ensure:

Stable recognition of a single frame.

Phase 3: All visible tiles

Add:

all rivers
all melds
dora
kita
Phase 4: Numeric / metadata perception

Complete:

scores
round
remaining tiles
honba
riichi sticks
Phase 5: Table-State Tracker

Without relying on operation buttons, recover and maintain from continuous table Observations:

ROUND_START
DRAW
DISCARD
RIICHI
MELD / CHI / PON / KAN
KITA
SCORE_UPDATE
ROUND_END
RESYNC

Sanma player identities must be established on physical screen slots locked for the entire match, rather than permanently binding player numbers to
bottom/left/right. At the start of the match, read the metadata one by one for the legal layouts in the configuration; only when the seat wind, dealer, round number, scores, and
pot conservation together support a unique candidate, and after repeated confirmation across consecutive stable frames, lock that match's `active_slots`.
Zero candidates or multiple candidates keep it unknown and break the confirmation streak; subsequent rounds reuse the locked layout, resetting only on a new match or an explicit
resync. Frames during layout confirmation should be cached and, after locking, sent back to state tracking in chronological order, so that no events are lost due to
pre-checks.

Phase 6: Interaction / Trigger Detection

Complete:

action button recognition
decision window detection
RON / TSUMO / GAME_END result evidence
DECISION_REQUIRED
AVAILABLE_ACTIONS_CHANGED
ACTION_WINDOW_OPEN / ACTION_WINDOW_CLOSE
legal-action exposure

`PASS` is only recorded as an AgentAction.

Phase 7: DeepSeek-Flash

Connect to the currently tested API.

Input:

MahjongState + legal_actions

Output:

DecisionResult

Phase 8: Replay evaluation

Run on the complete recording.

Must not click.

Phase 9: Executor

Add mouse control.

Priority:

dry run。

Phase 10: Verification loop

Execute:

action → visual verification。

Phase 11: Jev

JevBackend has completed the interface and real-call checks ahead of schedule per the user's request; this does not skip the visual acceptance of Phase 5/6.
When enabling Jev in the real-time pipeline, only switch the unified DecisionBackend configuration; do not rewrite other systems.

28. Acceptance Criteria

System completion must at least satisfy:

Perception

Ability to structurally recover from normal match screenshots:

round
scores
remaining tiles
honba
riichi sticks
dora
self hand
rivers
melds
riichi states
kita
available UI actions
State

Ability to correctly maintain across continuous recordings:

MahjongState

rather than understanding each frame independently.

Unknown / Active Learning

Persistent unknown will, after multi-frame retries, generate a deduplicated, reviewable case and enter HUMAN_REVIEW; manual labels
after confirmation via the local interactive GUI only enter the versioned verified dataset and training queue. Feishu is used for to-do reminders,
and labels are not required to be completed in chat. Any model upgrade must pass the golden regression gate.

Decision

Call the model only when necessary.

All model output is restricted to:

legal action set。

Backend

Current:

DeepSeek-Flash is running normally.

Future:

Switching to Jev only requires:

decision:
  primary: jev

Do not modify other modules.

Execution

Action execution must go through:

visual verification。

29. Engineering Constraints

Must comply with:

Do not hardcode specific situations in the current recording.

Error:

if hand == "...":
    discard("5p")
Do not write special cases to pass the current sample.

All logic must generalize.

UI-specific knowledge can enter:

roi.yaml
templates/
vision config

and should not be scattered throughout business logic.

Mahjong rule knowledge must be centralized in:

rules/

Model-specific logic must be centralized in:

decision/

Screen coordinate knowledge must be centralized in:

calibration/
executor/

Perception is not responsible for mahjong strategy.
The LLM is not responsible for visual recognition.
The LLM is not responsible for judging whether an action is legal.
The Executor is not responsible for decision-making.
30. First Concrete Task

First, do not implement the full Agent.

The first step is to complete:

Extract a representative reference frame from the test video.
Implement roi_editor.py.
Allow interactive mouse drawing and naming of ROIs.
Save the normalized roi.yaml.
Reload the ROIs in another frame.
Draw all ROIs back onto the original image.
Output roi_debug.png.
Export each crop separately.
Write a README explaining how to recalibrate the ROIs.
Do not start the decision model and mouse-click module until the ROI pipeline verification is complete.

After completion, report:

New files
ROI list
reference resolution
debug image
Existing uncertain regions
Suggestions for the next stage
