"""Local, hash-pinned recognition of already located numeric text."""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re
from threading import Condition, Thread
from time import monotonic

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class NumericOCRConfig:
    model: Path
    sha256: str
    minimum_confidence: float = .95
    maximum_digits: int = 3
    cache_size: int = 128
    asynchronous: bool = False
    retry_seconds: float = .5

    @classmethod
    def load(cls, path, root):
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or data.get('schema_version') != 1:
            raise ValueError('Unsupported numeric OCR configuration')
        model = Path(data['model'])
        value = cls(model if model.is_absolute() else root/model, str(data['sha256']),
                    float(data.get('minimum_confidence', .95)),
                    data.get('maximum_digits', 3), data.get('cache_size', 128),
                    data.get('asynchronous', False), float(data.get('retry_seconds', .5)))
        if (not re.fullmatch('[0-9a-f]{64}', value.sha256) or
                not math.isfinite(value.minimum_confidence) or
                not 0 < value.minimum_confidence <= 1 or
                type(value.maximum_digits) is not int or not 1 <= value.maximum_digits <= 6 or
                type(value.cache_size) is not int or not 0 <= value.cache_size <= 1024 or
                type(value.asynchronous) is not bool or
                not math.isfinite(value.retry_seconds) or not 0 < value.retry_seconds <= 10):
            raise ValueError('Invalid numeric OCR bounds')
        return value


class NumericOCR:
    def __init__(self, config, *, engine=None):
        self.config = config
        digest = sha256(config.model.read_bytes()).hexdigest()
        if digest != config.sha256:
            raise ValueError('Numeric OCR model hash mismatch')
        if engine is None:
            from rapidocr import RapidOCR, ModelType, OCRVersion, LangRec
            engine = RapidOCR(params={
                'Global.use_det': False, 'Global.use_cls': False,
                'Global.log_level': 'warning', 'Rec.lang_type': LangRec.EN,
                'Rec.model_type': ModelType.MOBILE, 'Rec.ocr_version': OCRVersion.PPOCRV4,
                'Rec.model_path': str(config.model),
                'EngineConfig.onnxruntime.intra_op_num_threads': 1,
                'EngineConfig.onnxruntime.inter_op_num_threads': 1})
            # Load and warm up before entering a match. Runtime never downloads weights.
            engine(np.zeros((48, 48, 3), np.uint8), use_det=False, use_cls=False)
        self.engine = engine
        self.cache = OrderedDict()
        self._condition = Condition()
        self._pending, self._latest, self._results = {}, {}, {}
        self._active = None
        self._thread = None
        self._async_enabled = False
        self._closed = False
        self.manifest = {'model': str(config.model), 'sha256': digest,
                         'minimum_confidence': config.minimum_confidence,
                         'maximum_digits': config.maximum_digits,
                         'device': 'CPU', 'threads': 1, 'only_recognition': True,
                         'asynchronous_after_layout_lock': config.asynchronous,
                         'retry_seconds': config.retry_seconds}

    def enable_async(self):
        self._async_enabled = self.config.asynchronous

    def read_for_field(self, patch, field):
        if not self._async_enabled:
            return self.read(patch)
        if field not in ('remaining_tiles', 'honba', 'riichi_sticks'):
            raise ValueError('Only noncritical numeric fields use async OCR')
        if not isinstance(patch, np.ndarray) or patch.ndim != 3 or not patch.size:
            return None, 0., 'numeric_ocr_empty_crop'
        key = (patch.shape, sha256(patch.tobytes()).digest())
        now = monotonic()
        with self._condition:
            if self._closed:
                return None, 0., 'numeric_ocr_closed'
            self._latest[field] = key
            previous = self._results.get(field)
            answer = previous[1] if previous and previous[0] == key else None
            retry = answer is None or answer[2] is not None and now-previous[2] >= self.config.retry_seconds
            if retry and self._active != (field, key) and self._pending.get(field, (None,))[0] != key:
                # At most one latest crop per field plus one active inference.
                # Pixel equality, not an old timestamp/value, permits reuse.
                self._pending[field] = (key, patch.copy())
                if self._thread is None:
                    self._thread = Thread(target=self._run_async, name='majsoul-numeric-ocr', daemon=True)
                    self._thread.start()
                self._condition.notify()
            return answer if answer is not None else (None, 0., 'numeric_ocr_pending')

    def _run_async(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._pending))
                if self._closed:
                    return
                field = next(iter(self._pending))
                key, patch = self._pending.pop(field)
                self._active = (field, key)
            try:
                answer = self.read(patch)
            except Exception:
                # An unexpected preprocessing error must not kill recovery for
                # all remaining fields in the bounded worker.
                answer = None, 0., 'numeric_ocr_failed'
            with self._condition:
                if self._latest.get(field) == key:
                    self._results[field] = (key, answer, monotonic())
                self._active = None
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(.2)

    def read(self, patch):
        if (not isinstance(patch, np.ndarray) or patch.ndim != 3 or
                patch.shape[2] != 3 or patch.size == 0):
            return None, 0., 'numeric_ocr_empty_crop'
        if np.ptp(patch) == 0:
            return None, 0., 'numeric_ocr_blank_crop'
        key = (patch.shape, sha256(patch.tobytes()).digest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        scale = 48/patch.shape[0]
        image = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        try:
            result = self.engine(image, use_det=False, use_cls=False)
            texts, scores = result.txts, result.scores
            text = texts[0] if texts is not None and len(texts) == 1 else ''
            score = float(scores[0]) if scores is not None and len(scores) == 1 else 0.
            if (not math.isfinite(score) or not self.config.minimum_confidence <= score <= 1 or
                    not re.fullmatch(r'[0-9]{1,'+str(self.config.maximum_digits)+'}', text)):
                answer = None, score, 'numeric_ocr_rejected'
            else:
                answer = text, score, None
        except Exception:
            answer = None, 0., 'numeric_ocr_failed'
        # Rejected crops must be retried; only accepted exact pixels are cached.
        if self.config.cache_size and answer[2] is None:
            self.cache[key] = answer
            if len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
        return answer
