"""The Agent dismisses reviewed in-game popups itself (no takeover process)."""
from pathlib import Path

from majsoul.calibration.frames import read_image
from majsoul.live.capture import LiveFrame
from majsoul.live.collector import CollectConfig
from majsoul.live.modals import ModalReader
from majsoul.live.session import AgentSession

ROOT = Path(__file__).resolve().parents[1]
AFK = ROOT/'artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000623.jpg'
TABLE = ROOT/'artifacts/live/protected-20261018-151224/client-002/frames/00000020.jpg'
MANIFEST = ROOT/'artifacts/templates/protection/manifest.json'


class Window:
    def __init__(self):
        self.clicks = []

    def click(self, point, **kwargs):
        self.clicks.append(point)


def make(tmp_path, image, *, execute=True):
    frame = LiveFrame(image, 0., (0, 0, image.shape[1], image.shape[0]), (0, 0))
    window, logs, notes = Window(), [], []
    session = AgentSession(window, CollectConfig.load(ROOT/'configs/collect.steam.yaml'), execute=execute,
                           capture=lambda: frame, sleep=lambda _: None,
                           log=lambda event, **fields: logs.append(event), directory=tmp_path,
                           notify=notes.append, modals=ModalReader(MANIFEST))
    return session, frame, window, logs, notes


def test_agent_clicks_a_reviewed_afk_popup_with_bounded_retries(tmp_path):
    session, frame, window, logs, notes = make(tmp_path, read_image(AFK))
    gap = session.config.confirm_click_seconds
    assert session.observe(frame, 0.) == (True, [])
    assert len(window.clicks) == 1 and logs[:2] == ['modal_observed', 'modal_confirm_clicked']
    session.observe(frame, gap/2)
    assert len(window.clicks) == 1  # no click storm while the popup animates out
    for step in range(1, 5):
        session.observe(frame, step*gap)
    assert len(window.clicks) == AgentSession.MODAL_ATTEMPTS
    assert any('did not close' in note for note in notes)


def test_dry_run_agent_never_clicks_a_popup(tmp_path):
    session, frame, window, logs, notes = make(tmp_path, read_image(AFK), execute=False)
    assert session.observe(frame, 0.) == (True, [])
    assert window.clicks == [] and 'modal_observed' in logs


def test_table_frame_is_not_a_popup(tmp_path):
    session, frame, window, logs, notes = make(tmp_path, read_image(TABLE))
    session.observe(frame, 0.)
    assert 'modal_observed' not in logs and window.clicks == []
