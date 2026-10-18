# Mahjong Soul DeepSeek annotation four-step script package

Run in the project root directory `D:\pScript\majsoul`. First install project dependencies and ensure the local config can read
`DEEPSEEK_API_KEY` or refer to `API_KEY` in the project's `../PuwuQuiz/.env`. Keys will not be written to reports.

DeepSeek classification by default forces direct connection and ignores system `HTTP_PROXY` / `HTTPS_PROXY`. Only when a proxy is explicitly needed
should `--proxy <URL>` be passed manually.

## Current 32,784 crops: start directly from step 3

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\03_classify_with_deepseek.py `
  datasets\raw\captured\captured_yonma_20260923_005221\tile_crops\manifest.json

.venv\Scripts\python tools\deepseek_annotation_pipeline\04_human_review.py `
  datasets\raw\captured\captured_yonma_20260923_005221\tile_crops\manifest.json `
  --reviewer human
```

Step 3 can be run repeatedly: successfully classified images are skipped, and only unfinished items are processed. It is recommended to first verify 32:

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\03_classify_with_deepseek.py `
  datasets\raw\captured\captured_yonma_20260923_005221\tile_crops\manifest.json `
  --maximum 32 `
  --report artifacts\deepseek-classification-smoke.json
```

The review interface only shows images that already have a final DeepSeek classification:

- `1`: Pass, confirm the model label; ordinary tiles are copied to `datasets/verified/tiles/<label>/` and enter the training queue.
- `2`: Reject the current DeepSeek label, then you must select the correct 37-class label or "invalid sample".
  Canceling the correction window will not write any result.
- When the DeepSeek label itself is `reject`, pressing `1` means confirming that the crop is invalid.
- `Alt+←/→`: browse only, do not write data.
- The pending queue is first sorted by `occurrence_count` from high to low; when occurrence counts are equal, then by
  "DeepSeek tile type × physical position (region/row-column/orientation)" balanced sorting.
- Click `Verified statistics` to view how many verified samples each of all 37 tile types already has (including 0).
- After the status switches to `verified`, select a misclicked sample and press `2` to reselect the correct tile type or invalidate it;
  derived files will be migrated synchronously, correction history is written to the SQLite audit table, and the raw crop will not be modified.

## Run all four steps from a new video

### 1. Video frame extraction

The frame extraction directory can be very large. Default JPEG 95, one frame every 15 frames, supports `--resume`.

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\01_extract_video_frames.py `
  "屏幕录制 2026-09-23 005221.mp4" `
  --source-manifest datasets\manifests\sources\captured_yonma_20260923_005221.json `
  --output-root datasets\raw\captured\captured_yonma_20260923_005221\frames `
  --stride 15
```

After interruption, append `--resume` with the original parameters to continue. The script will not overwrite an existing directory with different parameters.

### 2. ROI segmentation, tile cropping, and deduplication

Four-player example:

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\02_segment_roi_crops.py `
  datasets\raw\captured\captured_yonma_20260923_005221\frames\manifest.json `
  --output-root datasets\raw\captured\captured_yonma_20260923_005221\tile_crops `
  --active-slots bottom top left right --players 4 --no-kita
```

After interruption, append `--resume`. The output is always a raw corpus with `label=null, training_allowed=false`;
the split results or model results will not be used directly as training ground truth.

### 3. DeepSeek Flash single-pass classification

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\03_classify_with_deepseek.py `
  <tile_crops directory>\manifest.json `
  --batch-size 32 --concurrency 8 `
  --proxy http://127.0.0.1:7890
```

Each crop is fed into `deepseek-flash` as the original image. The model may only return one of the 37 tile classes or `reject`; the response must, one by one,
cover the request IDs, have valid confidence, and contain no extra fields; otherwise the batch is automatically split smaller and retried. The output only goes into
pending manual confirmation records in `datasets/manifests/corpus_reviews/<corpus-id>.sqlite3`.

### 4. Human two-key confirmation

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\04_human_review.py `
  <tile_crops directory>\manifest.json --reviewer <your name>
```

## Safety boundaries

- DeepSeek classification is not human ground truth, and will not be automatically written into verified or start training.
- The raw manifest and original crops are never modified.
- Only after pressing `1` in step 4 do normal tile labels enter verified and the pending-training queue.
- `player_count=4/mode=yonma` from four-player sources will be retained and must not be mixed in as sanma rule labels.
