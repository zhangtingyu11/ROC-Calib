"""Scene-wide independent feature inspection; no matching or calibration calls."""
import hashlib
import json
import time
import cv2
from datetime import datetime, timezone
from . import calibration as c
from . import calibration_review as r
from .lidar_structure import extract_intersections, EXTRACTION_VERSION
from .groups import _path, _write_json


def report_path(group, scene_key):
    root=_path(group)
    if not (root/'group.json').is_file():
        raise ValueError('任务组不存在')
    key=hashlib.sha256(scene_key.encode()).hexdigest()[:24]
    return root/'diagnostics'/'structure-extraction'/f'{key}.json'


def extract_scene(payload, progress):
    frames=payload.get('frames',[])
    if not frames or len(frames)>1000:
        raise ValueError('请选择含 1–1000 帧的场景进行提取检查')
    if len({f['datasetId'] for f in frames})!=1:
        raise ValueError('一次提取检查只处理当前场景，不混合多个场景')
    results=[]
    for i, frame in enumerate(frames):
        started=time.perf_counter()
        record={key:frame[key] for key in ('datasetId','frame','imageUrl','cloudUrl','width','height')}
        try:
            # Neither branch receives the user's extrinsic or cross-modal matches.
            image_payload={key:frame[key] for key in ('imageUrl','intrinsic','distortion','distortionModel','width','height')}
            image_payload.update(roi=[[0,0],[frame['width'],frame['height']]],lineLimit=None)
            image_lines=r.detect_lines(image_payload,lambda *_:None)['lines']
            points,_=c._read_cloud(c._resolve_cloud(frame['cloudUrl']))
            def local(percent,message,details=None):
                progress(int(100*(i+min(percent,90)/100)/len(frames)),f"{i+1}/{len(frames)} · 帧 {frame['frame']} · {message}",None)
            intersections,stats=extract_intersections(points,local,include_planes=True)
            planes=stats.pop('planes')
            lines=[{'id':j,'planeIds':item['planeIds'],'endpoints':item['fit']['endpoints'],
                    'spacingM':item['fit']['spacingM']} for j,item in enumerate(intersections)]
            record.update(status='ok',imageEdges=image_lines,planes=planes,intersections=lines,
                          stats={**stats,'imageEdgeCount':len(image_lines)})
        except (ValueError,OSError,KeyError,cv2.error) as error:
            record.update(status='error',error=str(error),imageEdges=[],planes=[],intersections=[])
        record['seconds']=round(time.perf_counter()-started,3)
        results.append(record)
        progress(int(100*(i+1)/len(frames)),f"完成 {i+1}/{len(frames)} 帧独立提取",None)
    result={'sceneKey':payload['sceneKey'],'version':EXTRACTION_VERSION,'mode':'independent-extraction-only',
            'generatedAt':datetime.now(timezone.utc).isoformat(),'frames':results,
            'frameCount':len(frames),'successfulFrames':sum(f['status']=='ok' for f in results),
            'note':'仅提取检查：没有跨模态匹配、配对创建或外参优化。'}
    if payload.get('group'):
        # Derived diagnostic cache only, separate from annotations and results.
        _write_json(report_path(payload['group'],payload['sceneKey']),result)
    return result


def load_report(group,scene_key):
    path=report_path(group,scene_key)
    return json.loads(path.read_text()) if path.is_file() else None
