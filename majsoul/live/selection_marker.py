"""Real UI marker evidence; never supplies an executable button or click point."""
from hashlib import sha256
import json
from pathlib import Path

import cv2

from ..calibration.frames import read_image

ROOT = Path(__file__).resolve().parents[2]


class SelectionCancelMarker:
    def __init__(self, config, *, root=ROOT):
        self.config = config
        self.template = None
        self.manifest = None
        if config.cancel_marker_manifest is None:
            return
        path = Path(root) / config.cancel_marker_manifest
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data.get('kind') != 'selection_cancel':
            raise ValueError('Invalid selection cancel marker manifest')
        size, box = data.get('source_size', ()), data.get('crop_box', ())
        if (len(size) != 2 or any(type(n) is not int or n <= 0 for n in size) or
                len(box) != 4 or any(type(n) is not int for n in box) or
                not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]) or
                data.get('player_count') not in (3,4) or data.get('mode') not in ('sanma','yonma')):
            raise ValueError('Cancel marker source geometry/provenance unavailable')
        template_path = Path(root) / data['template']
        if sha256(template_path.read_bytes()).hexdigest() != data['template_sha256']:
            raise ValueError('Cancel marker template hash changed')
        template = read_image(template_path)
        if template.shape[1::-1] != (box[2]-box[0], box[3]-box[1]):
            raise ValueError('Cancel marker crop dimensions changed')
        self.template = template
        self.source_size = tuple(size)
        self.manifest = {**data,'manifest_sha256':sha256(path.read_bytes()).hexdigest()}

    def read(self, frame, roi):
        if self.template is None or roi is None:
            return {'accepted':False,'reason':'cancel_marker_unavailable'}
        height, width = frame.image.shape[:2]
        if (width,height) != tuple(frame.window[2:]):
            return {'accepted':False,'reason':'cancel_capture_geometry_mismatch'}
        tw = round(self.template.shape[1]*width/self.source_size[0])
        th = round(self.template.shape[0]*height/self.source_size[1])
        if min(tw,th) < 3:
            return {'accepted':False,'reason':'cancel_marker_too_small'}
        template = cv2.resize(self.template,(tw,th),interpolation=cv2.INTER_AREA)
        boxes = roi.pixel_boxes((width,height))
        candidates = []
        possible = False
        for name in self.config.cancel_marker_regions:
            if name not in boxes:
                continue
            x1,y1,x2,y2 = boxes[name]
            crop = frame.image[y1:y2,x1:x2]
            if crop.shape[0] <= th or crop.shape[1] <= tw:
                continue
            scores = cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED)
            # At most two separated matches per region; neighbouring pixels of
            # one marker cannot be mistaken for two independent UI targets.
            for _ in range(2):
                _,score,_,(x,y) = cv2.minMaxLoc(scores)
                possible = possible or score >= self.config.cancel_marker_minimum_score
                clipped = x == 0 or y == 0 or x+tw == crop.shape[1] or y+th == crop.shape[0]
                if not clipped:
                    candidates.append((float(score),(x+x1,y+y1,x+x1+tw,y+y1+th),name))
                scores[max(0,y-th):y+th+1,max(0,x-tw):x+tw+1] = -1
        candidates.sort(reverse=True)
        unique = []
        for candidate in candidates:
            b = candidate[1]
            if not any(abs(b[0]-other[1][0]) < tw/2 and abs(b[1]-other[1][1]) < th/2
                       for other in unique):
                unique.append(candidate)
        if not unique:
            return {'accepted':False,'possible':possible,'reason':'cancel_marker_missing_or_clipped'}
        score,box,name = unique[0]
        second = unique[1][0] if len(unique)>1 else -1.
        accepted = score >= self.config.cancel_marker_minimum_score and score-second >= self.config.cancel_marker_minimum_margin
        return {'accepted':accepted,'possible':possible,'score':score,'margin':score-second,'box':box,'region':name,
                'manifest_sha256':self.manifest['manifest_sha256'],
                'reason':'real_cancel_marker' if accepted else 'cancel_marker_unrecognized_or_ambiguous'}
