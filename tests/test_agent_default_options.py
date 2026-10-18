from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.live.capture import LiveFrame
from majsoul.live.collector import CollectConfig
from majsoul.live.session import AgentSession
from majsoul.live.capture import FocusUnavailable
from majsoul.calibration.frames import read_image

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source', [
    'artifacts/live/protected-20261013-round01/client-002/frames/00000001.jpg',
    'artifacts/live/protected-20261013-round02/client-002/frames/00000001.jpg',
])
def test_matching_screens_suppress_tile_recognition_even_after_a_previous_table(tmp_path, source):
    config=CollectConfig.load(ROOT/'configs/collect.steam.yaml')
    image=read_image(ROOT/source)
    frame=LiveFrame(image,0.,(0,0,*image.shape[1::-1]),(0,0))
    session=AgentSession(None,config,execute=False,capture=lambda:frame,
        sleep=lambda _:None,log=lambda *a,**k:None,directory=tmp_path,
        notify=lambda *a,**k:False)
    assert session.classify(frame)==(False,None,False)
    assert session.observe(frame,0.)[0]
    session.lifecycle.update(confirm=False,table=True,lobby=False,now=1.)
    session.lifecycle.update(confirm=False,table=True,lobby=False,now=2.)
    assert not session.lifecycle.pending
    assert session.observe(frame,3.)[0]


@pytest.mark.parametrize('execute',[False,True])
def test_agent_first_and_next_hand_never_open_autoplay_drawer(tmp_path,execute):
    config=CollectConfig.load(ROOT/'configs/collect.steam.yaml')
    frame=LiveFrame(np.zeros((200,400,3),np.uint8),0.,(0,0,400,200),(0,0))
    def forbidden(*args,**kwargs):
        pytest.fail('Agent must not spend the turn checking default-OFF auto-play')
    session=AgentSession(SimpleNamespace(click=forbidden),config,execute=execute,
        capture=lambda:frame,sleep=forbidden,log=lambda *a,**k:None,
        directory=tmp_path,notify=lambda *a,**k:True)
    session.panel.ensure=forbidden
    session.ui=SimpleNamespace(match_result=lambda image:False)
    session.classify=lambda f:(False,None,True)
    assert session.observe(frame,0.)[0]
    assert not session.observe(frame,config.table_ready_seconds)[0]
    assert session.lifecycle.round_index==1
    session.classify=lambda f:(False,'blue',False)
    assert session.observe(frame,3.)[0]
    session.classify=lambda f:(False,None,True)
    assert session.observe(frame,4.)[0]
    assert not session.observe(frame,4.+config.table_ready_seconds)[0]
    assert session.lifecycle.round_index==2


@pytest.mark.parametrize('failures',[1,3])
def test_agent_result_focus_failure_reobserves_without_unbounded_retry(tmp_path,failures):
    config=CollectConfig.load(ROOT/'configs/collect.steam.yaml')
    frame=LiveFrame(np.zeros((200,400,3),np.uint8),0.,(0,0,400,200),(0,0))
    attempts=[]
    def click(*args,**kwargs):
        attempts.append(kwargs['expected'])
        if len(attempts)<=failures:
            raise FocusUnavailable('No input sent')
    captures=[]
    def capture():
        captures.append(frame)
        return frame
    session=AgentSession(SimpleNamespace(click=click),config,execute=True,
        capture=capture,sleep=lambda _:None,log=lambda *a,**k:None,
        directory=tmp_path,notify=lambda *a,**k:True)
    session.classify=lambda f:(False,'gold',False)
    session.ui=SimpleNamespace(match_result=lambda image:False,confirm=lambda image:'gold')
    session.observe(frame,0.)
    if failures>=config.confirm_focus_attempts:
        with pytest.raises(FocusUnavailable):
            for i in range(config.confirm_focus_attempts):
                session.observe(frame,1.+i*config.confirm_click_seconds)
        assert len(attempts)==config.confirm_focus_attempts
    else:
        session.observe(frame,1.)
        session.observe(frame,1.+config.confirm_click_seconds/2)
        assert len(attempts)==1
        session.observe(frame,1.+config.confirm_click_seconds)
        assert len(attempts)==2
    assert len(captures)==len(attempts)
