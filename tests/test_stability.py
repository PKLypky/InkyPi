"""Tests for the Pi Zero 2 stability fixes: lock scope, timeouts, watchdog, atomic config.

Run with: PYTHONPATH=src pytest
"""
import json
import os
import socket
import sys
import threading
import time
import types

import pytest
from PIL import Image

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC))

import refresh_task as rt  # noqa: E402
from model import RefreshInfo  # noqa: E402
from utils import image_utils, systemd_notify  # noqa: E402


# ---------------------------------------------------------------- refresh task

class FakeConfig:
    def __init__(self, **values):
        self.values = {"plugin_cycle_interval_seconds": 3600, "timezone": "UTC", **values}
        self.refresh_info = RefreshInfo(None, None, None, None)
        self.writes = 0

    def get_config(self, key=None, default=None):
        return self.values.get(key, default)

    def get_playlist_manager(self):
        return None

    def get_refresh_info(self):
        return self.refresh_info

    def get_plugin(self, plugin_id):
        return {"id": plugin_id}

    def write_config(self):
        self.writes += 1


class BlockingDisplay:
    """Display whose update blocks until `release` is set, like a slow ACeP refresh."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def display_image(self, image, image_settings=None):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=30)


class FakeAction(rt.RefreshAction):
    def execute(self, plugin, device_config, current_dt):
        return Image.new("RGB", (8, 8), "white")

    def get_refresh_info(self):
        return {"refresh_type": "Manual Update", "plugin_id": "fake"}

    def get_plugin_id(self):
        return "fake"


@pytest.fixture
def task(monkeypatch):
    monkeypatch.setattr(rt, "get_plugin_instance", lambda cfg: types.SimpleNamespace(config={}))
    created = []

    def make(config=None, display=None):
        t = rt.RefreshTask(config or FakeConfig(), display or BlockingDisplay())
        t.start()
        created.append(t)
        return t

    yield make
    for t in created:
        t.display_manager.release.set()
        t.stop()


def test_lock_is_free_while_display_updates(task):
    t = task()
    worker = threading.Thread(target=t.manual_update, args=(FakeAction(),), daemon=True)
    worker.start()
    assert t.display_manager.started.wait(5), "display update never started"

    # A web request that signals a config change must not wait for the panel.
    started = time.monotonic()
    t.signal_config_change()
    assert time.monotonic() - started < 0.5

    t.display_manager.release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert t.device_config.writes == 1


def test_manual_update_times_out_instead_of_hanging(task):
    t = task(config=FakeConfig(max_refresh_seconds=1))
    with pytest.raises(TimeoutError):
        t.manual_update(FakeAction())
    # the refresh is still stuck after its budget, so the watchdog must report unhealthy
    time.sleep(0.1)
    assert t.is_healthy() is False


def test_manual_update_queued_during_refresh_is_processed(task):
    t = task()
    first = threading.Thread(target=t.manual_update, args=(FakeAction(),), daemon=True)
    first.start()
    assert t.display_manager.started.wait(5)

    # Second request arrives while the first is still drawing (same image hash after the
    # first one, so only the first actually hits the display).
    second_done = threading.Event()

    def second():
        t.manual_update(FakeAction())
        second_done.set()

    threading.Thread(target=second, daemon=True).start()
    time.sleep(0.2)
    assert not second_done.is_set(), "second request returned before it was processed"

    t.display_manager.release.set()
    assert second_done.wait(5)
    assert t.device_config.writes == 2
    assert t.is_healthy()


def test_manual_update_propagates_plugin_errors(task):
    class Failing(FakeAction):
        def execute(self, plugin, device_config, current_dt):
            raise ValueError("boom")

    t = task()
    with pytest.raises(ValueError, match="boom"):
        t.manual_update(Failing())
    assert t.is_healthy()


# ---------------------------------------------------------------- chromium timeout

def test_screenshot_timeout_kills_whole_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    # parent spawns a grandchild (like Chromium's helper processes) and waits
    command = ["sh", "-c", f"sleep 60 & echo $! > {pid_file}; wait"]
    started = time.monotonic()
    returncode, _, _ = image_utils._run_with_timeout(command, 1)
    assert returncode is None
    assert time.monotonic() - started < 10

    child = int(pid_file.read_text())
    time.sleep(0.2)
    assert not _process_running(child), "grandchild survived the timeout"


def _process_running(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            state = next(line for line in f if line.startswith("State:"))
    except FileNotFoundError:
        return False
    return " Z " not in state and "zombie" not in state


def test_run_with_timeout_returns_output():
    returncode, stdout, _ = image_utils._run_with_timeout(["sh", "-c", "echo ok"], 5)
    assert returncode == 0
    assert stdout.strip() == b"ok"


# ---------------------------------------------------------------- e-paper busy timeout

def _waveshare_display(monkeypatch, config_values, epd):
    pytest.importorskip("display.waveshare_display")
    from display import waveshare_display as wd

    epdconfig = types.ModuleType("epdconfig")
    epdconfig.delay_ms = lambda ms: time.sleep(ms / 1000)
    epdconfig.exits = 0

    def module_exit():
        epdconfig.exits += 1

    epdconfig.module_exit = module_exit

    display = wd.WaveshareDisplay.__new__(wd.WaveshareDisplay)
    display.device_config = FakeConfig(**config_values)
    display.epd_display = epd(epdconfig)
    display.epd_display_init = lambda: None
    display.bi_color_display = False
    display._busy_deadline = None
    display._epdconfig = epdconfig
    wd._install_busy_timeout(epdconfig, display)
    return wd, display, epdconfig


class StuckEPD:
    """Driver whose BUSY pin never goes idle."""

    def __init__(self, epdconfig):
        self.epdconfig = epdconfig
        self.cleared = 0

    def getbuffer(self, image):
        return b""

    def Clear(self):
        self.cleared += 1

    def display(self, buf):
        while True:  # same shape as Waveshare ReadBusyH()
            self.epdconfig.delay_ms(5)

    def sleep(self):
        pass


class GoodEPD(StuckEPD):
    def display(self, buf):
        self.epdconfig.delay_ms(5)


def test_stuck_busy_pin_raises_and_powers_off(monkeypatch):
    wd, display, epdconfig = _waveshare_display(monkeypatch, {"display_timeout_seconds": 0.3}, StuckEPD)
    started = time.monotonic()
    with pytest.raises(wd.DisplayTimeoutError):
        display.display_image(Image.new("RGB", (8, 8)))
    assert time.monotonic() - started < 5
    assert epdconfig.exits == 1
    assert display._busy_deadline is None


def test_clear_is_skipped_by_default(monkeypatch):
    _, display, _ = _waveshare_display(monkeypatch, {}, GoodEPD)
    display.display_image(Image.new("RGB", (8, 8)))
    assert display.epd_display.cleared == 0


def test_clear_can_be_enabled(monkeypatch):
    _, display, _ = _waveshare_display(monkeypatch, {"clear_before_refresh": True}, GoodEPD)
    display.display_image(Image.new("RGB", (8, 8)))
    assert display.epd_display.cleared == 1


# ---------------------------------------------------------------- systemd notify

def test_notify_without_socket_is_noop(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert systemd_notify.notify("READY=1") is False


def test_notify_sends_datagram(monkeypatch, tmp_path):
    path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", path)
        assert systemd_notify.notify("READY=1") is True
        server.settimeout(1)
        assert server.recv(64) == b"READY=1"
    finally:
        server.close()


def test_watchdog_interval(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "180000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))
    assert systemd_notify.watchdog_interval_seconds() == 90
    monkeypatch.setenv("WATCHDOG_PID", "1")
    assert systemd_notify.watchdog_interval_seconds() is None
    monkeypatch.delenv("WATCHDOG_USEC")
    assert systemd_notify.watchdog_interval_seconds() is None


def test_watchdog_stops_heartbeat_when_unhealthy(monkeypatch):
    sent = []
    monkeypatch.setattr(systemd_notify, "watchdog_interval_seconds", lambda: 0.05)
    monkeypatch.setattr(systemd_notify, "notify", lambda msg: sent.append(msg) or True)
    healthy = {"value": True}
    systemd_notify.start_watchdog(lambda: healthy["value"])
    time.sleep(0.3)
    assert sent, "no heartbeat while healthy"
    healthy["value"] = False
    time.sleep(0.1)
    count = len(sent)
    time.sleep(0.3)
    assert len(sent) == count, "heartbeat continued while unhealthy"


# ---------------------------------------------------------------- config

def test_config_write_is_atomic(monkeypatch, tmp_path):
    from config import Config

    config_file = tmp_path / "device.json"
    config_file.write_text(json.dumps({"name": "InkyPi"}))
    monkeypatch.setattr(Config, "config_file", str(config_file))
    cfg = Config()
    cfg.update_value("name", "Test", write=True)

    assert json.loads(config_file.read_text())["name"] == "Test"
    assert not (tmp_path / "device.json.tmp").exists()

    # concurrent writers never produce a corrupt file
    threads = [threading.Thread(target=cfg.update_value, args=(f"k{i}", i, True)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    data = json.loads(config_file.read_text())
    assert all(f"k{i}" in data for i in range(20))
