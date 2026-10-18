"""Versioned, reviewed modal text/button anchors; no generic confirm guessing."""
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from ..calibration.frames import read_image


def _foreground_process_name(user):
    user.GetForegroundWindow.restype=wintypes.HWND
    user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    foreground=user.GetForegroundWindow()
    if not foreground:return None
    pid=wintypes.DWORD()
    user.GetWindowThreadProcessId(foreground,ctypes.byref(pid))
    if not pid.value:return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,
                                               wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    process=kernel.OpenProcess(0x1000,False,pid.value)  # QUERY_LIMITED_INFORMATION only.
    if not process:return None
    try:
        size=wintypes.DWORD(32768)
        name=ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(process,0,name,ctypes.byref(size)):return None
        return PureWindowsPath(name.value).name.casefold()
    finally:kernel.CloseHandle(process)


def input_desktop_status():
    """Separate a confirmed lock from an unavailable foreground observation."""
    user=ctypes.WinDLL('user32',use_last_error=True)
    user.OpenInputDesktop.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    user.OpenInputDesktop.restype=wintypes.HANDLE
    user.GetUserObjectInformationW.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    user.CloseDesktop.argtypes=[wintypes.HANDLE]
    desktop=user.OpenInputDesktop(0,False,1)
    if not desktop:
        return {'status':'unknown','reason':'open_input_desktop_failed',
                'winerror':ctypes.get_last_error()}
    try:
        name=ctypes.create_unicode_buffer(256);needed=wintypes.DWORD()
        if not user.GetUserObjectInformationW(desktop,2,name,ctypes.sizeof(name),ctypes.byref(needed)):
            return {'status':'unknown','reason':'desktop_name_failed',
                    'winerror':ctypes.get_last_error()}
        if name.value.lower()!='default':
            return {'status':'blocked','reason':'non_default_desktop','desktop':name.value}
        # OpenInputDesktop may still report Default while LockApp owns the
        # foreground. An unknown/protected foreground is also not permission
        # to send input; never try to activate or dismiss the lock screen.
        executable=_foreground_process_name(user)
        if executable is None:
            return {'status':'unknown','reason':'foreground_query_failed',
                    'winerror':ctypes.get_last_error()}
        return {'status':'blocked' if executable in ('lockapp.exe','logonui.exe') else 'available',
                'reason':'foreground_process','foreground_process':executable}
    finally:user.CloseDesktop(desktop)


def input_desktop_available():
    return input_desktop_status()['status']=='available'


class ModalReader:
    def __init__(self,manifest):
        self.entries={}
        if manifest is None:return
        manifest=Path(manifest)
        data=json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('schema_version')!=1:raise ValueError('Unsupported modal manifest')
        for entry in data['modals']:
            name=entry['name']
            if name in self.entries:raise ValueError('Duplicate modal name')
            anchors=[]
            for item in entry['anchors']:
                path=manifest.parent/item['file']
                if sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Modal template hash mismatch')
                region=item['region']
                if len(region)!=4 or not 0<=region[0]<region[2]<=1 or not 0<=region[1]<region[3]<=1:
                    raise ValueError('Invalid modal region')
                anchors.append((tuple(region),read_image(path)))
            if len(anchors)<2:raise ValueError('Modal needs text and button anchors')
            if not all(0<v<1 for v in entry['point']):raise ValueError('Invalid modal point')
            self.entries[name]=(entry,anchors)

    def read(self,image):
        height,width=image.shape[:2];matches=[]
        for name,(entry,anchors) in self.entries.items():
            accepted=True
            for region,template in anchors:
                x1,y1,x2,y2=(round(v*(width if i%2==0 else height)) for i,v in enumerate(region))
                patch=image[y1:y2,x1:x2]
                if not patch.size:accepted=False;break
                size=(min(256,template.shape[1]),min(64,template.shape[0]))
                patch=cv2.GaussianBlur(cv2.resize(patch,size),(3,3),0)
                reference=cv2.GaussianBlur(cv2.resize(template,size),(3,3),0)
                score=float(cv2.matchTemplate(patch,reference,cv2.TM_CCOEFF_NORMED)[0,0])
                error=float(np.abs(patch.astype(np.float32)-reference).mean()/255)
                if score<entry['minimum_score'] or error>entry['maximum_error']:
                    accepted=False;break
            if accepted:matches.append(name)
        return matches[0] if len(matches)==1 else None

    def point(self,name,size):
        entry,_=self.entries[name]
        return tuple(v*size[i] for i,v in enumerate(entry['point']))

    def recoverable(self,name):
        return self.entries[name][0].get('recoverable',True)
