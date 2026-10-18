from pathlib import Path
from types import SimpleNamespace
import cv2
import pytest

from majsoul.calibration.frames import read_image
from majsoul.live.modals import ModalReader

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def reader():return ModalReader(ROOT/'artifacts/templates/protection/manifest.json')


@pytest.mark.parametrize('source,name',[
    ('artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000623.jpg','afk'),
    ('artifacts/live/acceptance-20261010/agent-recheck-172934/operator-takeover-through-match/frames/00000018.jpg','connection'),
    ('artifacts/live/workflow-audit/acceptance-20261013/round10-current.png','table_connection'),
    ('artifacts/live/workflow-audit/recovery-live-20261011-172420/preflight.png','logout'),
])
@pytest.mark.parametrize('scale',[1,.75])
def test_reviewed_modal_needs_text_and_button_together(reader,source,name,scale):
    image=read_image(ROOT/source)
    image=cv2.resize(image,None,fx=scale,fy=scale)
    assert reader.read(image)==name
    assert reader.recoverable(name)==(name!='logout')
    height,width=image.shape[:2]
    image[round(.60*height):round(.70*height),round(.41*width):round(.59*width)]=0
    assert reader.read(image) is None


@pytest.mark.parametrize('source',[
    'artifacts/live/workflow-audit/tile-audit-20261011-172934/closure-lobby-2.png',
    'artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000411.jpg',
    'artifacts/live/workflow-audit/recovery-live-20261011-172420/after-connection-confirm.png',
    'artifacts/live/protected-20261013-round10/observer/000003.jpg',
])
def test_table_lobby_or_normal_final_confirm_are_not_modal_buttons(reader,source):
    assert reader.read(read_image(ROOT/source)) is None


@pytest.mark.parametrize('desktop,process,expected',[
    ('Default','lockapp.exe',False),('Default','logonui.exe',False),
    ('Default',None,False),('Winlogon','jantama_mahjongsoul.exe',False),
    ('Default','jantama_mahjongsoul.exe',True),('Default','chrome.exe',True),
])
def test_default_desktop_does_not_override_a_lock_or_unknown_foreground(monkeypatch,desktop,process,expected):
    import majsoul.live.modals as module
    closed=[]
    def name(_handle,_kind,buffer,*_args):
        buffer.value=desktop
        return 1
    user=SimpleNamespace(OpenInputDesktop=lambda *_:1,
                         GetUserObjectInformationW=name,CloseDesktop=lambda h:closed.append(h))
    monkeypatch.setattr(module.ctypes,'WinDLL',lambda *_a,**_k:user)
    monkeypatch.setattr(module,'_foreground_process_name',lambda _:process)
    assert module.input_desktop_available()==expected
    assert closed==[1]


@pytest.mark.parametrize('process,status',[(None,'unknown'),('lockapp.exe','blocked'),
                                           ('chrome.exe','available')])
def test_desktop_status_distinguishes_missing_foreground_from_lock(monkeypatch,process,status):
    import majsoul.live.modals as module
    def name(_handle,_kind,buffer,*_args):
        buffer.value='Default'
        return 1
    user=SimpleNamespace(OpenInputDesktop=lambda *_:1,
                         GetUserObjectInformationW=name,CloseDesktop=lambda _:None)
    monkeypatch.setattr(module.ctypes,'WinDLL',lambda *_a,**_k:user)
    monkeypatch.setattr(module,'_foreground_process_name',lambda _:process)
    assert module.input_desktop_status()['status']==status


def test_foreground_process_query_is_read_only_and_closes_its_handle(monkeypatch):
    import majsoul.live.modals as module
    calls=[]
    def pid(_hwnd,output):
        module.ctypes.cast(output,module.ctypes.POINTER(module.wintypes.DWORD)).contents.value=123
        return 456
    def query(_handle,_flags,buffer,_size):
        buffer.value=r'C:\Windows\SystemApps\Windows.LockApp\LockApp.exe'
        return 1
    def open_process(access,inherit,process_id):
        calls.append((access,inherit,process_id))
        return 9
    kernel=SimpleNamespace(OpenProcess=open_process,QueryFullProcessImageNameW=query,
                           CloseHandle=lambda handle:calls.append(('closed',handle)))
    user=SimpleNamespace(GetForegroundWindow=lambda:7,GetWindowThreadProcessId=pid)
    monkeypatch.setattr(module.ctypes,'WinDLL',lambda *_a,**_k:kernel)
    assert module._foreground_process_name(user)=='lockapp.exe'
    assert calls==[(0x1000,False,123),('closed',9)]
