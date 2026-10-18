from types import SimpleNamespace

import pytest

from majsoul.window import WindowsClient,WindowError
from majsoul.live.config import ExecutionConfig


@pytest.mark.parametrize('interrupted',[False,True])
def test_mouse_press_spans_polling_frames_and_always_releases(monkeypatch,interrupted):
    events=[]
    client=WindowsClient.__new__(WindowsClient)
    client.api=SimpleNamespace(SetCursorPos=lambda *p:True,
        mouse_event=lambda flag,*args:events.append(flag))
    client.check=lambda expected:None
    def hold(seconds):
        events.append(seconds)
        if interrupted:
            raise KeyboardInterrupt
    monkeypatch.setattr('majsoul.window.time.sleep',hold)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            client.click((10,10),(0,0,100,100),hold_seconds=.05)
    else:
        client.click((10,10),(0,0,100,100),hold_seconds=.05)
    assert events==[0x0002,.05,0x0004]


@pytest.mark.parametrize('value',[0,-.1,True,.3,float('nan')])
def test_invalid_mouse_press_is_rejected_before_input(value):
    client=WindowsClient.__new__(WindowsClient)
    with pytest.raises(WindowError):
        client.click((10,10),(0,0,100,100),hold_seconds=value)
    with pytest.raises(ValueError):
        ExecutionConfig.from_dict({'mouse_press_seconds':value})


def test_double_click_has_two_released_presses_and_rechecks_focus(monkeypatch):
    events=[]
    client=WindowsClient.__new__(WindowsClient)
    client.api=SimpleNamespace(SetCursorPos=lambda *p:True,
        mouse_event=lambda flag,*args:events.append(flag))
    client.check=lambda expected:events.append('check')
    monkeypatch.setattr('majsoul.window.time.sleep',lambda seconds:events.append(seconds))
    client.click((10,10),(0,0,100,100),click_count=2)
    assert events==['check','check',2,.05,4,.05,'check',2,.05,4]


def test_double_click_stops_before_second_press_if_focus_lost(monkeypatch):
    events=[]
    checks=[0]
    client=WindowsClient.__new__(WindowsClient)
    client.api=SimpleNamespace(SetCursorPos=lambda *p:True,
        mouse_event=lambda flag,*args:events.append(flag))
    def check(expected):
        checks[0]+=1
        if checks[0]==3:
            raise WindowError('focus lost')
    client.check=check
    monkeypatch.setattr('majsoul.window.time.sleep',lambda _:None)
    with pytest.raises(WindowError,match='focus lost'):
        client.click((10,10),(0,0,100,100),click_count=2)
    assert events==[2,4]


@pytest.mark.parametrize('value',[0,3,True,1.5])
def test_invalid_click_count_sends_no_input(value):
    with pytest.raises(WindowError):
        WindowsClient.__new__(WindowsClient).click((10,10),(0,0,100,100),click_count=value)
    with pytest.raises(ValueError):
        ExecutionConfig.from_dict({'hand_click_count':value})
