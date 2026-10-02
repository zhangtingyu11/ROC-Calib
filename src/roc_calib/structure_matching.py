"""Experimental, opt-in geometric proposals and joint object/line fitting.

No annotations or poses are written here. Every proposal needs human confirmation.
LiDAR surfaces are extracted independently from all finite returns; camera data
only enters subsequent matching. Fixed proposal gates are not accuracy guarantees.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from . import calibration as c
from . import calibration_review as r


def project(xyz, pose, k):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    uv = camera[:, :2] / np.maximum(camera[:, 2, None], .1)
    return uv * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]], camera[:, 2]


def automatic(payload, progress):
    if payload.get('temporalProjection'):
        raise ValueError('自动棱线暂不支持滚动快门；原物体配对不受影响')
    from .lidar_structure import extract_intersections, EXTRACTION_VERSION
    # The extraction call has no access to pose, image, or camera intrinsics.
    points, _ = c._read_cloud(c._resolve_cloud(payload['cloudUrl']))
    geometry, stats = extract_intersections(points, progress)
    images = r.detect_lines({**payload, 'lineLimit': None,
                             'roi': [[0, 0], [payload['width'], payload['height']]]}, progress)['lines']
    stats['imageEdgeCount'] = len(images)
    pose = r.matrix(payload['matrix'])
    k = np.asarray(payload['intrinsic'], float).reshape(3, 3)
    d = np.asarray(payload['distortion'], float)
    candidates = []
    for number, structure in enumerate(geometry):
        progress(60 + int(30 * number / max(1, len(geometry))), '用已有外参匹配两种模态的独立线特征', None)
        line_uv, depth = project(np.asarray(structure['fit']['endpoints']), pose, k)
        if np.any(depth <= .1) or not np.isfinite(line_uv).all():
            continue
        direction = line_uv[1] - line_uv[0]
        norm = np.linalg.norm(direction)
        if norm < 5:
            continue
        # Pose is used only here, after LiDAR planes/intersections are fixed.
        spacing = max(1., float(k[0,0] * structure['fit']['spacingM'] / np.mean(depth)))
        matches = []
        for image_index, image in enumerate(images):
            edge = r.undistort(image['imageLine'], k, d, payload['distortionModel'])
            axis = edge[1] - edge[0]
            length = np.linalg.norm(axis)
            if length < 15:
                continue
            axis /= length
            alignment = abs(float(direction @ axis / norm))
            error = abs((line_uv-edge[0]) @ np.array([-axis[1],axis[0]]))
            span = (line_uv-edge[0]) @ axis
            overlap = max(0.,min(length,span.max())-max(0.,span.min()))
            if alignment < .95 or error.max() > min(24.,max(4.,3*spacing)) or overlap < .3*min(length,norm):
                continue
            score = float(error.mean()/spacing+1-alignment)
            matches.append((score,image_index,image,float(error.mean()),alignment))
        if not matches:
            continue
        matches.sort(key=lambda item:item[0])
        score,image_index,image,error,alignment = matches[0]
        ambiguous = len(matches)>1 and matches[1][0]-score<.2
        item = {**payload, **structure, 'mode':'planes', 'imageLine':image['imageLine'],
                'curve':image['curve'],'origin':'automatic','confirmed':False,'enabled':False,
                'proposalPose':payload['matrix'],'score':score,'ambiguous':ambiguous,
                'extractionVersion':EXTRACTION_VERSION,
                'evidence':{'projectionMeanPx':error,'directionCosine':alignment,
                            'supportCount':len(structure['indicesA'])+len(structure['indicesB']),
                            'spacingPx':spacing,'alternativeCount':len(matches)-1},
                'inputVersion':r.fingerprint(payload),'imageEdgeIndex':image_index}
        candidates.append(item)
    # Flag shared image edges too; no automatic confirmation of ambiguous matches.
    for item in candidates:
        item['ambiguous'] |= sum(other['imageEdgeIndex']==item['imageEdgeIndex'] for other in candidates)>1
    stats['matchCount']=len(candidates)
    return {'candidates':sorted(candidates,key=lambda item:item['score']),'extraction':stats,
            'note':'图像边缘与点云双平面交线独立提取；已有外参仅用于匹配。所有配对仍需人工确认。'}


def prepare_lines(structures, base, progress):
    lines = []
    for item in structures:
        if not item.get('confirmed') or item.get('enabled') is False:
            continue
        if item.get('temporalProjection'):
            raise ValueError('联合棱线暂不支持滚动快门')
        fit = r.fit_structure(item, progress)
        k = np.asarray(item['intrinsic'], float).reshape(3, 3)
        edge = r.undistort(item['imageLine'], k, np.asarray(item['distortion'], float), item['distortionModel'])
        line = np.cross(np.r_[edge[0], 1], np.r_[edge[1], 1])
        if np.linalg.norm(line[:2]) < 2:
            raise ValueError('图像棱线过短')
        support, depth = project(np.asarray(fit['points']), base, k)
        support = support[(depth > .1) & np.isfinite(support).all(axis=1)]
        if len(support) < 2:
            raise ValueError('棱线有效投影支撑不足')
        spacing = max(1., float(np.median(cKDTree(support).query(support, k=2)[0][:, 1])))
        lines.append((np.linspace(*np.asarray(fit['endpoints']), 5), k, line/np.linalg.norm(line[:2]), spacing))
    return lines


def line_costs(lines, pose):
    costs, pixels = [], []
    for xyz, k, line, spacing in lines:
        uv, depth = project(xyz, pose, k)
        error = abs(uv @ line[:2] + line[2])
        error = np.where(depth > .1, error, 1e4 + np.maximum(0, .1-depth)*1e3)
        # Match the object's local-spacing units; equal vote per correspondence.
        costs.append(float(np.mean((error/spacing)**2)))
        pixels.append(float(np.mean(error)))
    return costs, pixels


def joint(payload, progress):
    base = r.matrix(payload['matrix'])
    _, pairs = r.prepared_request(payload['request'], base)
    lines = prepare_lines(payload['structures'], base, progress)
    if not lines or not pairs:
        raise ValueError('需要已确认的棱线和物体配对才能联合优化')
    def metrics(pose):
        objects = r.evaluate(pairs, pose)
        costs, pixels = line_costs(lines, pose)
        total = float(np.mean(objects['pairCosts'] + costs))
        return {'jointCost': total, 'objects': objects, 'lineMeanPx': float(np.mean(pixels))}
    before = metrics(base)
    scale = np.r_[np.ones(3), np.full(3, max(1., np.median([np.linalg.norm(p['points'],axis=1).mean() for p in pairs])))]
    def residual(delta):
        pose=c._delta_matrix(delta*scale) @ base
        values=[np.sqrt(max(0., v)) for v in r.evaluate(pairs,pose)['pairCosts']]
        for xyz,k,line,spacing in lines:
            uv,depth=project(xyz,pose,k)
            error=(uv @ line[:2]+line[2])/spacing
            values.extend(np.where(depth>.1,error,1e4))
        return np.asarray(values)
    eps=1e-6
    jac=np.column_stack([(residual(np.eye(6)[i]*eps)-residual(-np.eye(6)[i]*eps))/(2*eps) for i in range(6)])
    singular=np.linalg.svd(jac,compute_uv=False)
    if len(singular)<6 or singular[0]<=0 or np.count_nonzero(singular>singular[0]*1e-6)<6:
        raise ValueError('当前联合约束的局部秩不足，请补充不同位置或方向的配对；保留原外参')
    progress(20, '联合优化物体区域、轮廓与已确认棱线', None)
    fit = minimize(lambda delta: metrics(c._delta_matrix(delta*scale) @ base)['jointCost'],
                   np.zeros(6), method='Powell', options={'maxiter': 60, 'maxfev': 5000})
    candidate = c._delta_matrix(fit.x*scale) @ base
    after = metrics(candidate)
    accepted = bool(fit.success and np.isfinite(after['jointCost']) and after['jointCost'] < before['jointCost'])
    return {'matrix': (candidate if accepted else base).ravel().tolist(), 'before': before, 'after': after,
            'recommended': accepted, 'converged': bool(fit.success), 'experimental': True,
            'objectCount': len(pairs), 'structureCount': len(lines), 'inputVersion': r.fingerprint(payload),
            'localConstraintRank': 6,
            'note': '全部配对共同求解；残差降低不代表真值精度提高。确认应用才保存新版本。'}
