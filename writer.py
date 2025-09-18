# writer.py
import time
import threading
from queue import Queue, Empty

from database import get_connection

_WRITE_QUEUE: "Queue[tuple]" = Queue()
_WORKER_STARTED = False

def _worker():
    """Background worker: executes queued SQL writes one-by-one."""
    while True:
        try:
            query, params, result_slot = _WRITE_QUEUE.get(timeout=1)
        except Empty:
            continue

        ok, err = False, None
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            conn.close()
            ok = True
        except Exception as e:
            err = str(e)

        # notify the enqueuer
        result_slot["ok"] = ok
        result_slot["err"] = err
        _WRITE_QUEUE.task_done()

def start_writer():
    """Start the single writer thread once."""
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    _WORKER_STARTED = True

def enqueue_write(query: str, params: tuple = (), timeout: float = 10.0):
    """
    Queue a write and wait until it's done (or times out).

    Returns:
        (ok, rows, wait_log, eta_seconds, err)
    """
    # Rough ETA: queue size * 0.4s (assume ~400ms per write on phone)
    qlen_before = _WRITE_QUEUE.qsize()
    eta = max(1, min(int(qlen_before * 0.4) + 1, 30))

    result_slot = {"ok": False, "err": None}
    _WRITE_QUEUE.put((query, params, result_slot))

    waited = 0.0
    wait_log = [f"queued (pos≈{qlen_before + 1})"]
    tick = 0.2  # poll every 200ms

    while waited < timeout:
        if result_slot["ok"] or result_slot["err"]:
            break
        time.sleep(tick)
        waited += tick
        # every second, append a countdown-ish message
        if int(waited) > len([m for m in wait_log if m.startswith("⏳")]):
            remaining = max(0, eta - int(waited))
            wait_log.append(f"⏳ {remaining}s…")

    ok = bool(result_slot["ok"])
    err = result_slot["err"]
    return ok, None, wait_log, eta, err
