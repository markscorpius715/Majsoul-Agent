from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.perception.numeric_ocr import NumericOCR, NumericOCRConfig


def reader(tmp_path, text='46', score=.99, *, fail=False, cache_size=2):
    model = tmp_path/'model.onnx'
    model.write_bytes(b'test-only-injected-engine')
    calls = []

    def engine(image, **kwargs):
        calls.append(image)
        if fail:
            raise RuntimeError('inference failed')
        return SimpleNamespace(txts=[text], scores=[score])

    result = NumericOCR(NumericOCRConfig(model, sha256(model.read_bytes()).hexdigest(),
                                        cache_size=cache_size), engine=engine)
    return result, calls


def patch():
    image = np.zeros((25, 40, 3), np.uint8)
    image[4:20, 5:15] = 255
    return image


@pytest.mark.parametrize('text,score', [('4O', .999), ('', .999),
                                     ('46', .94), ('46', float('nan')), ('46', 1.1), ('1000', .999)])
def test_rejects_non_digits_low_confidence_and_invalid_sequences(tmp_path, text, score):
    ocr, _ = reader(tmp_path, text, score)
    value, _, error = ocr.read(patch())
    assert value is None
    assert error == 'numeric_ocr_rejected'


@pytest.mark.parametrize('text', ['09', '08', '00', '046'])
def test_zero_padded_ui_digits_remain_recognized(tmp_path, text):
    ocr, _ = reader(tmp_path, text)
    assert ocr.read(patch()) == (text, .99, None)


def test_blank_and_failure_never_produce_a_number(tmp_path):
    ocr, calls = reader(tmp_path, fail=True)
    assert ocr.read(np.zeros((25, 40, 3), np.uint8))[0] is None
    assert not calls
    assert ocr.read(patch()) == (None, 0., 'numeric_ocr_failed')
    assert ocr.read(patch()) == (None, 0., 'numeric_ocr_failed')
    assert len(calls) == 2  # Transient inference errors cannot poison the cache.


def test_cache_requires_identical_pixels_and_is_bounded(tmp_path):
    ocr, calls = reader(tmp_path)
    image = patch()
    assert ocr.read(image) == ('46', .99, None)
    assert ocr.read(image.copy()) == ('46', .99, None)
    assert len(calls) == 1
    for value in range(1, 4):
        changed = image.copy()
        changed[0, 0] = value
        ocr.read(changed)
    assert len(ocr.cache) == 2
    ocr.read(image)
    assert len(calls) == 5


def test_missing_or_wrong_model_cannot_silently_fallback(tmp_path):
    config = NumericOCRConfig(tmp_path/'absent.onnx', '0'*64)
    with pytest.raises(FileNotFoundError):
        NumericOCR(config, engine=object())
    config.model.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        NumericOCR(config, engine=object())


@pytest.mark.parametrize('text,score,accepted', [('46', .99, True), ('46', .90, False),
                                              ('4', .99, False), ('4O', .99, False)])
def test_real_metadata_rejection_does_not_fallback_to_template(tmp_path, text, score, accepted):
    from pathlib import Path
    from majsoul.calibration.frames import read_image
    from majsoul.calibration.roi_config import ROIConfig
    from majsoul.live.config import LiveConfig
    from majsoul.perception import VisionConfig, GameMetadataReader, GlyphTemplateClassifier, extract_roi
    from majsoul.perception.metadata_reader import _NumericSpec
    root = Path(__file__).resolve().parents[1]
    config = LiveConfig.load(root/'configs/live.steam.yaml')
    vision = VisionConfig.load(config.vision)
    ocr, calls = reader(tmp_path, text, score)
    metadata = GameMetadataReader(vision, GlyphTemplateClassifier(config.glyph_templates, vision),
                                  read_scores=False, numeric_ocr=ocr)
    image = read_image(root/'artifacts/live/acceptance-20261010/agent-recheck-131424/diagnostics/000096.jpg')
    crop = extract_roi(image, ROIConfig.load(config.roi), 'remaining_tiles')
    reading = metadata._read_numeric(crop, _NumericSpec('center', 'prefix_remaining'))
    assert len(calls) == 1
    assert reading.complete == accepted
    assert reading.value == (46 if accepted else None)
    # The old template can read 46 now; rejected OCR must still leave it unknown.
    if not accepted:
        assert reading.issues
    crop.image[:] = 0
    blank = metadata._read_numeric(crop, _NumericSpec('center', 'prefix_remaining'))
    assert not blank.complete and blank.value is None
    assert len(calls) == 1
