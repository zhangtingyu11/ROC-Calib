"""Opt-in review and structural constraints; never mutates calibration inputs."""
from __future__ import annotations

import hashlib
import json
import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from . import calibration as c


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def matrix(value):
    result = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.isfinite(result).all() or not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError('外参矩阵无效')
    if np.linalg.det(result[:3,:3]) <= 0 or not np.allclose(result[:3,:3].T @ result[:3,:3], np.eye(3), atol=1e-3):
        raise ValueError('旋转矩阵无效')
    return result


def selected_points(url, indices):
    cloud, _ = c._read_cloud(c._resolve_cloud(url))
    ids = np.unique(np.asarray(indices, dtype=np.int64))
    if np.any(ids < 0) or np.any(ids >= len(cloud)):
        raise ValueError('点索引已失效，请重新载入原始帧')
    valid = np.isfinite(cloud[ids]).all(axis=1)
    return cloud[ids][valid].astype(float), ids[valid]


def prepared_request(payload, pose):
    request = c.CalibrationRequest.model_validate(payload)
    pairs = []
    for source in request.pairs:
        # Review retains all selected valid returns, including original IDs.
        points, ids = selected_points(source.cloud_url, source.point_indices)
        pair = c._prepare_pair(source.model_copy(update={'point_indices':ids.tolist()}))
        pair['points'] = points
        pair['source_indices'] = ids
        pair['weight'] = 1.
        spacing = c._contour_sampling_spacings(pair, pose)
        pair['contour_sampling_spacings'] = spacing
        usable = spacing[np.isfinite(spacing) & (spacing > 0)]
        pair['objective_spacing'] = max(1., float(np.median(usable))) if len(usable) else 1.
        weights, _, info = c._projected_boundary_reliability(pair, pose)
        pair['boundary_weights'] = weights
        pair['projected_contour_reliability'] = info
        pairs.append(pair)
    return request, pairs


def isolated_indices(points):
    if len(points) < 6:
        return np.empty(0, dtype=int)
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(5, len(points)))
    spacing = np.maximum(distances[:, 1], 1e-6)
    # Sparse kNN connectivity: no quadratic dense distance matrix.
    rows = np.repeat(np.arange(len(points)), neighbors.shape[1]-1)
    cols = neighbors[:, 1:].ravel()
    lengths = distances[:, 1:].ravel()
    keep = lengths <= 3 * np.minimum(spacing[rows], spacing[cols])
    graph = coo_matrix((np.ones(np.count_nonzero(keep)), (rows[keep], cols[keep])), shape=(len(points),len(points)))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    main = counts.argmax()
    small = (counts[labels] <= max(2, int(len(points)*.03))) & (labels != main)
    separation = cKDTree(points[labels == main]).query(points)[0]
    return np.flatnonzero(small & (separation > 6 * np.median(spacing)))


def evaluate(pairs, pose):
    values = [c._joint_pair_cost(c._pair_terms(p, pose, include_diagnostics=False, objective_only=True)) for p in pairs]
    return {'meanCost': float(np.mean(values)) if values else None, 'pairCosts': values}


def review(payload, progress):
    pose = matrix(payload['matrix'])
    request, pairs = prepared_request(payload['request'], pose)
    result = []
    image_edges = {}
    for n, (source, pair) in enumerate(zip(request.pairs, pairs)):
        progress(5+int(85*n/len(pairs)), f'复查配对 {n+1}/{len(pairs)}', None)
        xy = c._project_all_points(pair, pair['points'], pose)
        finite = np.isfinite(xy).all(axis=1)
        outside = np.ones(len(xy), dtype=bool)
        outside[finite] = [not c._inside_mask(pair['mask'], p) for p in xy[finite]]
        errors = np.full(len(xy), math.hypot(*pair['mask'].shape))
        errors[finite] = c._sample_distance(pair['outside_distance'], xy[finite])
        # Exact off-frame distance, not a clamped-to-border success.
        h,w = pair['mask'].shape
        off = finite & ((xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h))
        errors[off] = np.hypot(xy[off,0]-np.clip(xy[off,0],0,w-1), xy[off,1]-np.clip(xy[off,1],0,h-1))
        isolated = isolated_indices(pair['points'])
        boundary = pair['boundary_points']
        _, bidx = cKDTree(pair['points']).query(boundary)
        low = bidx[np.asarray(pair['boundary_weights']) < .5]
        spacing = pair['objective_spacing']
        score = float(np.mean(np.log1p((errors/spacing)**2)/2))
        reasons = []
        edge_median = None
        raw_source = payload['request']['pairs'][n]
        image_url = raw_source.get('image_url')
        if image_url:
            if image_url not in image_edges:
                image = cv2.imread(str(c._resolve_cloud(image_url)), cv2.IMREAD_GRAYSCALE)
                image_edges[image_url] = (c.distance_transform_edt(cv2.Canny(image,50,150)==0)
                    if image is not None and image.shape==pair['mask'].shape else None)
            edge_map = image_edges[image_url]
            if edge_map is not None:
                edge_median = float(np.median(c._sample_distance(edge_map,pair['closed_edge_points'])))
                if edge_median>max(2.,spacing):
                    reasons.append('Mask 边缘附近缺少对应图像梯度：请对照原图检查漏分/多分；弱纹理和阴影也会触发此提示。')
        if len(isolated): reasons.append('存在远离主体的小点簇：检查是否多选背景；先定位，不要直接删除。')
        if np.count_nonzero(outside): reasons.append('高亮点落在 Mask 外：对照原图检查漏分或多选；小距离越界也可能来自采样和投影误差。')
        if len(low): reasons.append('部分边界缺少可靠支持：可能扫描不完整，不要补造点或强拉图像轮廓。')
        direction = None
        finite_outside = finite & outside & ~off
        if np.any(finite_outside):
            edge = pair['closed_edge_points']
            _, nearest = cKDTree(edge).query(xy[finite_outside])
            shift = np.median(xy[finite_outside]-edge[nearest], axis=0)
            if np.linalg.norm(shift)>1: direction = (shift/np.linalg.norm(shift)).tolist()
        result.append({'annotationId':source.annotation_id, 'score':score,
            'maskContours':[contour.reshape(-1,2).tolist() for contour in cv2.findContours(pair['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]],
            'outsideRatio':float(outside.mean()), 'outsideMeanPx':float(errors.mean()), 'spacingPx':spacing,
            'maskImageEdgeMedianPx':edge_median,
            'outsideIndices':pair['source_indices'][outside].tolist(),
            'isolatedIndices':pair['source_indices'][isolated].tolist(),
            'lowSupportIndices':np.unique(pair['source_indices'][low]).tolist(),
            'projection':[{'index':int(i),'xy':p.tolist()} for i,p in zip(pair['source_indices'][finite],xy[finite])],
            'reasons':reasons, 'direction':direction,
            'status':'观测不足' if len(low)>len(boundary)/2 else '未发现明显异常'})
    scores = np.array([x['score'] for x in result]); median = float(np.median(scores))
    mad = float(np.median(abs(scores-median))); threshold = median+3*max(1.4826*mad,1e-6)
    for item in result:
        if item['score']>threshold or item['isolatedIndices']:
            item['status']='建议复查'
        item['outlierThreshold']=threshold
    directions = [x['direction'] for x in result if x['direction'] is not None]
    warning = None
    if len(directions)>=3 and np.linalg.norm(np.mean(directions,axis=0))>.8:
        warning='多组配对呈一致方向偏移：先检查外参和相机模型，不建议逐个修改标注。'
    return {'inputVersion':fingerprint(payload), 'pairs':result, 'warning':warning}


def influence(payload, progress):
    pose = matrix(payload['matrix'])
    request, pairs = prepared_request(payload['request'], pose)
    keep = [p for p in request.pairs if p.annotation_id != payload['excludeId']]
    if len(keep)==len(request.pairs): raise ValueError('未找到指定配对')
    if len(keep)<4: raise ValueError('排除后不足四组配对，无法进行相同初始化试算')
    candidate = c.optimize(request.model_copy(update={'pairs':keep,'optimization_batch_id':None}), progress)
    new = matrix(candidate['matrix'])
    before, after = evaluate(pairs,pose), evaluate(pairs,new)
    excluded = next(i for i,p in enumerate(pairs) if p['id']==payload['excludeId'])
    rotation = float(np.degrees(np.linalg.norm(cv2.Rodrigues(new[:3,:3]@pose[:3,:3].T)[0])))
    shift = float(np.linalg.norm(new[:3,3]-pose[:3,3]))
    remaining = [i for i in range(len(pairs)) if i!=excluded]
    remaining_before=float(np.mean(np.array(before['pairCosts'])[remaining]))
    remaining_after=float(np.mean(np.array(after['pairCosts'])[remaining]))
    drift=[]
    for pair in pairs:
        oldxy=c._project_all_points(pair,pair['points'],pose);newxy=c._project_all_points(pair,pair['points'],new)
        valid=np.isfinite(oldxy).all(axis=1)&np.isfinite(newxy).all(axis=1)
        if np.any(valid): drift.append(float(np.median(np.linalg.norm(oldxy[valid]-newxy[valid],axis=1)/pair['objective_spacing'])))
    conflict = np.mean(np.array(after['pairCosts'])[remaining]) < np.mean(np.array(before['pairCosts'])[remaining]) and after['pairCosts'][excluded]>before['pairCosts'][excluded]
    return {'inputVersion':fingerprint(payload), 'matrix':candidate['matrix'], 'before':before,'after':after,
        'excludeId':payload['excludeId'],'remainingBefore':remaining_before,'remainingAfter':remaining_after,
        'excludedBefore':before['pairCosts'][excluded],'excludedAfter':after['pairCosts'][excluded],
        'pairIds':[p['id'] for p in pairs], 'rotationChangeDeg':rotation,'translationChangeM':shift,
        'conclusion':'与其他配对存在冲突' if conflict else ('影响较大' if drift and np.median(drift)>1 else '证据不足'),
        'note':'这是配对影响诊断，不是真值精度；不自动删除或应用。',
        'excludedProjection':c._project_all_points(pairs[excluded],pairs[excluded]['points'],new).tolist()}


def robust_fit(points, plane=False):
    if len(points)<(6 if plane else 4): raise ValueError('有效支撑点不足')
    weights = np.ones(len(points))
    for _ in range(20):
        center = np.average(points,axis=0,weights=weights)
        _, singular, axes = np.linalg.svd((points-center)*np.sqrt(weights[:,None]),full_matrices=False)
        direction = axes[-1 if plane else 0]
        residual = abs((points-center)@direction) if plane else np.linalg.norm(np.cross(points-center,direction),axis=1)
        scale = max(float(np.median(residual))*1.4826, 1e-6)
        weights = 1/np.sqrt(1+(residual/scale)**2)
    if plane and singular[1]<singular[0]*.03: raise ValueError('所选点近似共线，不能确定平面')
    if plane and singular[2]>singular[1]*.25: raise ValueError('所选点不是可靠平面，请调整支撑区域')
    if not plane and singular[1]>singular[0]*.2: raise ValueError('所选点没有清晰线状结构，请缩小点带或改用两表面')
    return center,direction,residual,scale


def fit_structure(payload, progress):
    a, aid = selected_points(payload['cloudUrl'],payload['indicesA'])
    if payload.get('extractionVersion') == 'independent-planes-v1' and payload['mode']=='planes':
        from .lidar_structure import observed_intersection
        b,bid=selected_points(payload['cloudUrl'],payload['indicesB'])
        if np.intersect1d(aid,bid).size: raise ValueError('两个表面的选点重叠')
        return {**observed_intersection(a,b,aid,bid),'inputVersion':fingerprint(payload)}
    plane = payload['mode']=='planes'
    ca, da, ra, sa = robust_fit(a,plane)
    support = [{'indices':aid.tolist(),'residualM':float(np.median(ra)), 'unsupportedIndices':aid[ra>3*sa].tolist()}]
    if plane:
        b,bid = selected_points(payload['cloudUrl'],payload['indicesB'])
        if np.intersect1d(aid,bid).size: raise ValueError('两个表面的选点重叠，请分别选择')
        cb,db,rb,sb = robust_fit(b,True)
        axis = np.cross(da,db); norm=np.linalg.norm(axis)
        if norm<.1: raise ValueError('两个表面近乎平行，无法可靠求交线')
        axis /= norm
        origin = np.linalg.lstsq(np.stack([da,db]),np.array([da@ca,db@cb]),rcond=None)[0]
        ta,tb=(a-origin)@axis,(b-origin)@axis
        lo,hi=max(ta.min(),tb.min()),min(ta.max(),tb.max())
        support.append({'indices':bid.tolist(),'residualM':float(np.median(rb)), 'unsupportedIndices':bid[rb>3*sb].tolist()})
        # Reject an intersection far outside either observed face.
        for pts in [a,b]:
            reach=np.min(np.linalg.norm(np.cross(pts-origin,axis),axis=1))
            spacing=np.median(cKDTree(pts).query(pts,k=2)[0][:,1])
            if reach>3*max(spacing,1e-6): raise ValueError('交线远离观测支撑，请选择靠近真实棱线的区域')
    else:
        origin,axis=ca,da; t=(a-origin)@axis;lo,hi=t.min(),t.max()
    if hi-lo<1e-4: raise ValueError('两侧观测没有共同的有效棱线范围')
    return {'endpoints':[(origin+lo*axis).tolist(),(origin+hi*axis).tolist()], 'support':support,
        'points':a.tolist()+(b.tolist() if plane else []),
        'label':'拟合几何线（非新增激光回波）','inputVersion':fingerprint(payload)}


def undistort(points,k,d,model):
    points=np.asarray(points,dtype=float).reshape(-1,1,2)
    if not np.isfinite(points).all(): raise ValueError('图像端点无效')
    if model=='fisheye': return cv2.fisheye.undistortPoints(points,k,d[:4],P=k).reshape(-1,2)
    return cv2.undistortPoints(points,k,d,P=k).reshape(-1,2)


def distort(points,k,d,model):
    rays=np.column_stack([(np.asarray(points)[:,0]-k[0,2])/k[0,0],(np.asarray(points)[:,1]-k[1,2])/k[1,1],np.ones(len(points))])
    if model=='fisheye': return cv2.fisheye.projectPoints(rays.reshape(1,-1,3),np.zeros(3),np.zeros(3),k,d[:4])[0].reshape(-1,2)
    return cv2.projectPoints(rays,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)


def detect_lines(payload, progress):
    image=cv2.imread(str(c._resolve_cloud(payload['imageUrl'])))
    if image is None: raise ValueError('原始图像不可用')
    k=np.array(payload['intrinsic'],dtype=float).reshape(3,3); d=np.array(payload['distortion'],dtype=float)
    model=payload['distortionModel']; h,w=image.shape[:2]
    if model=='fisheye': image=cv2.fisheye.undistortImage(image,k,d[:4],Knew=k)
    else: image=cv2.undistort(image,k,d,None,k)
    rect=np.asarray(payload['roi'],dtype=float).reshape(2,2)
    # ROI is specified in original pixels; filter candidates by inverse-mapped midpoint.
    lo,hi=rect.min(axis=0),rect.max(axis=0)
    lines=cv2.createLineSegmentDetector().detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))[0]
    candidates=[]
    for line in ([] if lines is None else lines):
        endpoints=line.reshape(2,2).astype(float)
        original=distort(np.linspace(endpoints[0],endpoints[1],24),k,d,model)
        if np.all(original>=lo) and np.all(original<=hi):
            candidates.append({'imageLine':original[[0,-1]].tolist(),'curve':original.tolist(),'length':float(np.linalg.norm(endpoints[1]-endpoints[0]))})
    candidates.sort(key=lambda x:-x['length'])
    limit=payload.get('lineLimit',30)
    return {'lines':candidates if limit is None else candidates[:int(limit)], 'note':'图像边缘线候选，需与独立提取的三维交线确认对应。'}


def line_curve(payload, progress):
    k=np.array(payload['intrinsic'],dtype=float).reshape(3,3); d=np.array(payload['distortion'],dtype=float)
    uv=undistort(payload['imageLine'],k,d,payload['distortionModel'])
    return {'curve':distort(np.linspace(uv[0],uv[1],40),k,d,payload['distortionModel']).tolist()}


def refine_structure(payload, progress):
    base=matrix(payload['matrix']); structures=payload['structures']
    if len(structures)<3: raise ValueError('至少需要三条已确认棱线，且位置与方向需提供独立约束')
    lines=[]
    for item in structures:
        if not item.get('confirmed'): raise ValueError('存在未确认的结构对应')
        if item.get('temporalProjection'): raise ValueError('结构修正暂不支持滚动快门模型；原标定功能不受影响')
        # Refit from original selections, never trust client-supplied 3D endpoints.
        fit=fit_structure(item,progress)
        xyz=np.linspace(*np.asarray(fit['endpoints']),5)
        k=np.array(item['intrinsic'],dtype=float).reshape(3,3);d=np.array(item['distortion'],dtype=float)
        uv=undistort(item['imageLine'],k,d,item['distortionModel'])
        line=np.cross(np.r_[uv[0],1],np.r_[uv[1],1]); norm=np.linalg.norm(line[:2])
        if norm<2: raise ValueError('图像棱线过短')
        lines.append((xyz,k,line/norm))
    scale=np.r_[np.ones(3),np.full(3,max(1.,np.median([np.linalg.norm(x[0],axis=1).mean() for x in lines])))]
    def residual(delta):
        pose=c._delta_matrix(delta*scale)@base
        values=[]
        for xyz,k,line in lines:
            cam=xyz@pose[:3,:3].T+pose[:3,3]
            uv=cam[:,:2]/np.maximum(cam[:,2,None],.1)
            pixels=uv*np.array([k[0,0],k[1,1]])+np.array([k[0,2],k[1,2]])
            error=pixels@line[:2]+line[2]
            error=np.where(cam[:,2]>.1,error,1e4+np.maximum(0,.1-cam[:,2])*1e3)
            values.extend(error)
        return np.asarray(values)
    zero=np.zeros(6); eps=1e-6
    def rank_at(delta):
        jac=np.column_stack([(residual(delta+np.eye(6)[i]*eps)-residual(delta-np.eye(6)[i]*eps))/(2*eps) for i in range(6)])
        s=np.linalg.svd(jac,compute_uv=False)
        return int(np.count_nonzero(s>s[0]*1e-6)) if s[0]>0 else 0
    if rank_at(zero)<6: raise ValueError('棱线约束退化：请补充不同位置和方向的结构')
    progress(20,'正在试算独立棱线修正',None)
    fit=least_squares(residual,zero,loss='soft_l1',f_scale=1.,max_nfev=200)
    if not fit.success or rank_at(fit.x)<6: raise ValueError('棱线修正未收敛或约束退化，保留原外参')
    candidate=c._delta_matrix(fit.x*scale)@base
    _,pairs=prepared_request(payload['request'],base)
    before,after=evaluate(pairs,base),evaluate(pairs,candidate)
    first=float(np.mean(np.abs(residual(zero))));last=float(np.mean(np.abs(residual(fit.x))))
    recommend=last<first and after['meanCost']<=before['meanCost']+1e-9
    return {'matrix':candidate.ravel().tolist(),'before':before,'after':after,'lineBeforePx':first,'lineAfterPx':last,
        'recommended':recommend,'inputVersion':fingerprint(payload),'experimental':True,
        'note':'候选仅供比较，需主动应用；残差降低不代表真值精度提高。'}


def dispatch(action,payload,progress):
    from .structure_matching import automatic, joint
    from .structure_extraction import extract_scene
    result={'review':review,'influence':influence,'fit':fit_structure,'detect':detect_lines,'refine':refine_structure,'curve':line_curve,'match':automatic,'joint':joint,'extract-scene':extract_scene}[action](payload,progress)
    # Invalid projections are represented as null, never nonstandard JSON NaN.
    def clean(value):
        if isinstance(value,float) and not math.isfinite(value): return None
        if isinstance(value,list): return [clean(v) for v in value]
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        return value
    return clean(result)
