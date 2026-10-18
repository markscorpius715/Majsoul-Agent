"""One-time reviewed UI template extraction, not model training."""
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from majsoul.calibration.frames import read_image,write_image

def main():
    out=ROOT/'artifacts/templates/protection'
    out.mkdir(parents=True,exist_ok=True)
    sources=[('afk','artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000623.jpg',
              [( .305,.440,.670,.518),(.419,.615,.585,.690)]),
             ('connection','artifacts/live/acceptance-20261010/agent-recheck-172934/operator-takeover-through-match/frames/00000018.jpg',
              [(.398,.455,.602,.505),(.416,.608,.585,.685)]),
             ('table_connection','artifacts/live/workflow-audit/acceptance-20261013/round10-current.png',
              [(.400,.455,.600,.510),(.419,.615,.585,.685)]),
             ('logout','artifacts/live/workflow-audit/recovery-live-20261011-172420/preflight.png',
              [(.314,.445,.675,.526),(.423,.615,.590,.690)])]
    entries=[]
    for name,source,regions in sources:
        image=read_image(ROOT/source);height,width=image.shape[:2]
        entry={'name':name,'point':[.5,.65],'minimum_score':.96,'maximum_error':.06,
               'recoverable':name!='logout',
               'source':source,'source_sha256':sha256((ROOT/source).read_bytes()).hexdigest(),
               'player_count':3,'mode':'sanma_east','anchors':[]}
        for index,region in enumerate(regions):
            x1,y1,x2,y2=(round(v*(width if i%2==0 else height)) for i,v in enumerate(region))
            path=out/f'{name}-{index}.png'
            write_image(path,image[y1:y2,x1:x2])
            entry['anchors'].append({'file':path.name,'region':region,'sha256':sha256(path.read_bytes()).hexdigest()})
        entries.append(entry)
    (out/'manifest.json').write_text(json.dumps({'schema_version':1,'purpose':'reviewed_modal_recovery',
        'modals':entries},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote reviewed text/button anchors for AFK and connection overlays; no game input')

if __name__=='__main__':main()
