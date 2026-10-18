"""Small protection-only UI reader; no tile-face model or public metadata OCR."""
import json
from ..calibration.frames import read_image
from ..calibration.roi_config import ROIConfig
from ..perception.config import VisionConfig
from ..perception.roi_extractor import extract_roi
from ..perception.tile_segmenter import HandSegmenter
from .capture import roi_for_frame
from .collector import CollectConfig,CollectionUI,menu_score
from .buttons import ActionButtonReader
from .countdown import CountdownReader
from .modals import ModalReader


class ProtectionUI:
    def __init__(self,config):
        self.config=config
        self.roi=ROIConfig.load(config.roi)
        self.collect=CollectConfig.load(config.lifecycle_ui)
        self.ui=CollectionUI(self.collect)
        self.lobby_template=read_image(self.collect.menu_template)
        self.buttons=ActionButtonReader(config.buttons,config.button_templates)
        self.clock=CountdownReader.load(config.countdown,config.rules.parents[1],json.loads(config.rules.read_text(encoding='utf-8')))
        self.hand=HandSegmenter(VisionConfig.load(config.vision))
        self.modals=ModalReader(config.protection.modal_manifest)

    def read(self,frame):
        image=frame.image
        roi=roi_for_frame(self.roi,image.shape[1::-1],self.config.projection)
        modal=self.modals.read(image)
        lobby=not modal and menu_score(image,self.lobby_template,self.collect.menu_region,self.collect.calibration_size)>=self.collect.menu_threshold
        confirm=None if modal else self.ui.confirm(image)
        table=not lobby and not confirm and self.ui.table_visible(image)
        reading=self.clock.read(image,roi) if self.clock else None
        labels=self.buttons.read(image,roi) if table and not modal else {}
        segmentation=self.hand.segment(extract_roi(image,roi,'self_hand')) if table and not modal else None
        count=len(segmentation.tiles) if segmentation else 0
        clean=segmentation is not None and not segmentation.issues
        # A transient extra face during dealing/sorting is not proof of a
        # live discard window. A separated drawn tile supplies local geometry;
        # calls without that geometry remain covered by the UI uncertainty
        # watchdog until an actual clock/button is visible.
        drawn_index=segmentation.drawn_index if segmentation else None
        own_turn=clean and count in (14,11,8,5,2) and drawn_index is not None
        total=reading.total_seconds if reading else None
        clock_visible=reading is not None and reading.status=='complete'
        actionable=table and not modal and (bool(labels) or own_turn or clock_visible)
        waiting=table and not modal and clean and count in (13,10,7,4,1) and not labels and not clock_visible
        return {'captured_at':frame.captured_at,'lobby':lobby,'confirm':confirm,'table':table,
            'modal':modal,'actionable':actionable,'waiting':waiting,
            'uncertain':not lobby and not confirm and not modal and not (actionable or waiting),
            'total_seconds':total,'hand_count':count,'drawn_index':drawn_index,
            'clock_status':reading.status if reading else None,
            'clock_reason':reading.reason if reading else None,'buttons':sorted(labels)}
