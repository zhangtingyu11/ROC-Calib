"""Bounded CPU-process isolation for concurrent camera-model solves.

Only plain request/result data crosses the pipe; SAM and its CUDA context stay
in the web process. Each worker exits after its solve, releasing geometry.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time

_SLOTS = threading.BoundedSemaphore(max(1, min(3, os.cpu_count() or 1)))


def _solve_in_child(payload: dict, sender, action=None) -> None:
    try:
        def progress(percent, message, details=None):
            sender.send(("progress", percent, message, details))
        if action is not None:
            from .calibration_review import dispatch
            sender.send(("result", dispatch(action, payload, progress)))
            return
        from .calibration import CalibrationRequest
        from .region_solver import optimize
        # Preparation caches cannot be shared across process address spaces.
        payload = {**payload, "optimization_batch_id": None, "optimization_batch_size": 1}
        request = CalibrationRequest.model_validate(payload)
        def progress(percent, message, details=None):
            sender.send(("progress", percent, message, details))
        sender.send(("result", optimize(request, progress)))
    except Exception as error:
        sender.send(("error", str(error)))
    finally:
        sender.close()


def optimize_isolated(request, progress=None, *, action=None, cancelled=None) -> dict:
    if progress:
        progress(1, "正在分配独立 CPU 求解进程", None)
    acquired = False
    try:
        while not acquired:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Task cancelled")
            acquired = _SLOTS.acquire(timeout=.2)
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError('任务已取消')
        # Never fork a web process that may already own a CUDA context.
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        payload = request if isinstance(request, dict) else request.model_dump()
        process = context.Process(target=_solve_in_child, args=(payload, sender, action))
        try:
            process.start()
            sender.close()
            deadline = time.monotonic() + float(os.environ.get("ROC_CALIB_SOLVER_TIMEOUT", "1800"))
            while True:
                if time.monotonic() > deadline:
                    raise RuntimeError("Solver timed out")
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError('任务已取消')
                if not receiver.poll(.1):
                    if not process.is_alive():
                        raise RuntimeError('标定求解进程异常退出，请重试')
                    continue
                try:
                    event = receiver.recv()
                except EOFError as error:
                    raise RuntimeError("标定求解进程异常退出，请重试") from error
                if event[0] == "result":
                    return event[1]
                if event[0] == "error":
                    raise RuntimeError(event[1])
                if progress:
                    progress(*event[1:])
        finally:
            receiver.close()
            sender.close()
            if process.pid is not None:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)

    finally:
        if acquired:
            _SLOTS.release()
