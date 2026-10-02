from __future__ import annotations
import asyncio
import json
import threading
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .calibration_worker import optimize_isolated
from .groups import _path, _write_json

router = APIRouter(prefix='/v1/calibration/review')
_STORE_LOCK = threading.RLock()


class StructureStore(BaseModel):
    revision: int = Field(ge=0)
    structures: list[dict] = Field(max_length=500)


def store_path(group):
    root = _path(group)
    if not (root/'group.json').exists(): raise HTTPException(404,'任务组不存在')
    return root/'annotations'/'structures.json'


@router.get('/structures/{group}')
def read_structures(group: str):
    try:
        path=store_path(group)
        return json.loads(path.read_text()) if path.exists() else {'revision':0,'structures':[]}
    except ValueError as error: raise HTTPException(400,str(error)) from error


@router.put('/structures/{group}')
def write_structures(group: str, body: StructureStore):
    with _STORE_LOCK:
        current=read_structures(group)
        if current['revision']!=body.revision: raise HTTPException(409,'结构标注已在其他页面更新，请重新载入')
        value={'revision':body.revision+1,'structures':body.structures}
        _write_json(store_path(group),value)
        return value


@router.get('/extraction/{group}')
def read_extraction(group: str, scene_key: str):
    from .structure_extraction import load_report
    try:
        result=load_report(group,scene_key)
        if result is None: raise HTTPException(404,'当前场景尚未提取')
        return result
    except ValueError as error: raise HTTPException(400,str(error)) from error


@router.post('/{action}')
async def run_review(action: Literal['review','influence','fit','detect','refine','curve','match','joint','extract-scene'], body: dict, request: Request):
    async def stream():
        loop=asyncio.get_running_loop(); events=asyncio.Queue(); cancelled=threading.Event()
        def put(value): loop.call_soon_threadsafe(events.put_nowait,value)
        def progress(percent,message,details=None): put({'type':'progress','percent':percent,'message':message})
        def work():
            try:
                result=optimize_isolated(body,progress,action=action,cancelled=cancelled)
                put({'type':'result','result':result})
            except Exception as error: put({'type':'error','message':str(error)})
        task=asyncio.create_task(asyncio.to_thread(work))
        try:
            while True:
                if await request.is_disconnected(): break
                try: event=await asyncio.wait_for(events.get(),timeout=.5)
                except asyncio.TimeoutError: continue
                yield json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'
                if event['type'] in {'result','error'}: break
        finally:
            cancelled.set()
            # Do not abandon the thread: it reaps its child after cancellation.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'X-Accel-Buffering':'no'})
