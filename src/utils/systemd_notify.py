"""Minimal sd_notify client and watchdog heartbeat for running under systemd.

The service runs with Type=notify and WatchdogSec. A background thread sends WATCHDOG=1
only while the refresh task reports itself healthy. If the refresh thread dies or a refresh
hangs (Chromium stuck, e-paper BUSY pin never released), heartbeats stop and systemd
restarts the service instead of leaving the display frozen forever.

Without NOTIFY_SOCKET (development, tests) every call is a no-op.
"""
import logging
import os
import socket
import threading

logger = logging.getLogger(__name__)


def notify(message: str) -> bool:
    """Send a raw sd_notify message, e.g. "READY=1". Returns True when sent."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]  # abstract namespace socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
        return True
    except OSError as e:
        logger.warning(f"sd_notify failed: {e}")
        return False


def watchdog_interval_seconds():
    """Half of the watchdog timeout systemd gave us, or None when the watchdog is off."""
    usec = os.environ.get("WATCHDOG_USEC")
    pid = os.environ.get("WATCHDOG_PID")
    if not usec or (pid and int(pid) != os.getpid()):
        return None
    try:
        return max(int(usec) / 1_000_000 / 2, 1.0)
    except ValueError:
        return None


def start_watchdog(is_healthy) -> threading.Thread | None:
    """Start a daemon thread that pings the systemd watchdog while `is_healthy()` is True."""
    interval = watchdog_interval_seconds()
    if interval is None:
        logger.info("systemd watchdog not enabled, heartbeat thread not started")
        return None

    def _loop():
        reported = False
        stop = threading.Event()
        while not stop.wait(interval):
            try:
                healthy = is_healthy()
            except Exception:
                logger.exception("Watchdog health check failed")
                healthy = False
            if healthy:
                notify("WATCHDOG=1")
                reported = False
            elif not reported:
                logger.error("Refresh task is hung or dead, stopping watchdog heartbeat so systemd restarts the service")
                reported = True

    thread = threading.Thread(target=_loop, name="systemd-watchdog", daemon=True)
    thread.start()
    logger.info(f"systemd watchdog heartbeat every {interval:.0f}s")
    return thread
