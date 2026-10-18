"""Offline numeric OCR study on real Steam ROIs; no game input or deployment."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from majsoul.calibration.frames import read_image,write_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.config import LiveConfig
from majsoul.live.countdown import CountdownConfig,CountdownReader
from majsoul.perception import VisionConfig,extract_roi
from majsoul.perception.metadata_reader import GlyphSegmenter

OUT=ROOT/'artifacts/numeric-ocr-study-20261010'
SOURCES=[
 'agent-recheck-131424/frames/00001-before.png',
 'agent-recheck-131424/frames/00001-after.png',
 'agent-recheck-131424/frames/00002-before.png',
 'agent-recheck-131424/frames/00002-after.png',
 'agent-recheck-131424/diagnostics/000096.jpg',
 'agent-recheck-123649/afk-after.png',
 'agent-recheck-120456/frames/00001-before.png',
 'agent-recheck-120456/frames/00001-after.png',
 'agent-recheck-125735/frames/00001-before.png',
 'agent-recheck-125735/frames/00001-after.png',
]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    c=LiveConfig.load(ROOT/'configs/live.steam.yaml');v=VisionConfig.load(c.vision)
    roi=ROIConfig.load(c.roi);segmenter=GlyphSegmenter(v)
    rules=json.loads(c.rules.read_text(encoding='utf-8'))
    clock=CountdownReader(CountdownConfig.load(c.countdown,ROOT),
                          turn_seconds=rules['turn_seconds'],reserve_seconds=rules['reserve_seconds'])
    items=[];cells=[]
    for name in SOURCES:
        source=ROOT/'artifacts/live/acceptance-20261010'/name
        image=read_image(source)
        for region,style in [('remaining_tiles','center'),('honba','counter'),('riichi_sticks','counter'),('countdown',None)]:
            if style:
                glyphs=segmenter.segment(extract_roi(image,roi,region),style)
                boxes=[asdict(g.box) for g in glyphs[1:]]
            else:
                reading=clock.read(image,roi)
                boxes=[g['box'] for g in reading.to_dict()['glyphs']]
            if not boxes:
                continue
            x1,y1=min(b['x1'] for b in boxes),min(b['y1'] for b in boxes)
            x2,y2=max(b['x2'] for b in boxes),max(b['y2'] for b in boxes)
            patch=image[int(y1):int(y2),int(x1):int(x2)]
            path=OUT/'crops'/f'{len(items):03d}.png'
            write_image(path,patch)
            items.append({'id':len(items),'source':str(source.relative_to(ROOT)),
                'source_sha256':sha256(source.read_bytes()).hexdigest(),'region':region,
                'box':[x1,y1,x2,y2],'crop':str(path.relative_to(OUT)),
                'expected':None,'label_source':'pending assistant visual review',
                'player_count':3,'mode':'sanma_east'})
            cell=np.full((110,260,3),245,np.uint8)
            cv2.putText(cell,f'{len(items)-1:02d} {region}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
            scale=min(248/patch.shape[1],75/patch.shape[0])
            show=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            cell[28:28+show.shape[0],4:4+show.shape[1]]=show
            cells.append(cell)
    while len(cells)%4:cells.append(np.full((110,260,3),245,np.uint8))
    write_image(OUT/'contact.png',np.vstack([np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)]))
    (OUT/'samples.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'samples':len(items),'contact':str(OUT/'contact.png')}))


def benchmark():
    import rapidocr,requests,yaml
    from rapidocr import RapidOCR,ModelType,OCRVersion,LangRec
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    from importlib.metadata import version
    config=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    registry_path=Path(rapidocr.__file__).parent/'default_models.yaml'
    registry=yaml.safe_load(registry_path.read_text(encoding='utf-8'))
    spec=registry['onnxruntime']['PP-OCRv4']['rec']['en_PP-OCRv4_rec_mobile']
    model=OUT/'en_PP-OCRv4_rec_mobile.onnx'
    if not model.exists():
        proxy=config.decision.jev_proxy
        response=requests.get(spec['model_dir'],proxies={'http':proxy,'https':proxy},timeout=60)
        response.raise_for_status()
        model.write_bytes(response.content)
    digest=sha256(model.read_bytes()).hexdigest()
    assert digest==spec['SHA256'],'Official model hash mismatch'
    engine=RapidOCR(params={'Global.use_det':False,'Global.use_cls':False,
        'Global.log_level':'warning','Rec.lang_type':LangRec.EN,'Rec.model_type':ModelType.MOBILE,
        'Rec.ocr_version':OCRVersion.PPOCRV4,'Rec.model_path':str(model),
        'EngineConfig.onnxruntime.intra_op_num_threads':1,
        'EngineConfig.onnxruntime.inter_op_num_threads':1})
    samples=json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    assert all(s['expected'] is not None for s in samples),'Review labels first'
    engine(read_image(OUT/samples[0]['crop']),use_det=False,use_cls=False)
    rows=[];prepared=[]
    for sample in samples:
        patch=read_image(OUT/sample['crop'])
        # One uniform preprocessing for all samples; do not pick per-image output.
        scale=48/patch.shape[0]
        patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        patch=cv2.copyMakeBorder(patch,4,4,8,8,cv2.BORDER_CONSTANT,value=(0,0,0))
        prepared.append(patch)
        times=[];result=None
        for _ in range(5):
            start=time.perf_counter()
            result=engine(patch,use_det=False,use_cls=False)
            times.append(time.perf_counter()-start)
        text=result.txts[0] if result.txts else ''
        score=float(result.scores[0]) if result.scores else None
        rows.append({**sample,'predicted':text,'score':score,
            'correct':text==sample['expected'],'mean_seconds':sum(times)/len(times),'max_seconds':max(times)})
    batch_times=[]
    batch_samples=prepared[:4]
    for _ in range(5):
        start=time.perf_counter()
        batch_result=engine.text_rec(TextRecInput(img=batch_samples))
        batch_times.append(time.perf_counter()-start)
    assert list(batch_result.txts)==[r['predicted'] for r in rows[:4]],'Batch output differs'
    report={'model_url':spec['model_dir'],'model_sha256':digest,'model_bytes':model.stat().st_size,
        'rapidocr_version':version('rapidocr'),'device':'CPU','threads':1,
        'only_recognition':True,'cold_load_excluded':True,
        'preprocessing':'original BGR crop, cubic resize to height48, black border y4/x8',
        'samples':len(rows),'correct':sum(r['correct'] for r in rows),
        'mean_seconds':sum(r['mean_seconds'] for r in rows)/len(rows),
        'batch_four_mean_seconds':sum(batch_times)/len(batch_times),
        'batch_four_max_seconds':max(batch_times),
        'scope':'development samples with assistant visual labels; not independent gold acceptance',
        'results':rows}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False))
    print(json.dumps([{'id':r['id'],'expected':r['expected'],'predicted':r['predicted'],'score':r['score']} for r in rows if not r['correct']]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','benchmark'])
    args=parser.parse_args()
    prepare() if args.mode=='prepare' else benchmark()
