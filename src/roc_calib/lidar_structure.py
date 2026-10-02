"""LiDAR-only planar patches and observed finite intersections.

This module accepts XYZ and original indices, never camera data or an extrinsic.
Neighborhoods and covariance are evaluated for every finite return (no thinning).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

EXTRACTION_VERSION = 'independent-planes-v1'


def _runs(values, gap):
    values = np.sort(values)
    return [(part[0], part[-1]) for part in np.split(values, np.flatnonzero(np.diff(values)>gap)+1)
            if len(part)>=3 and part[-1]>part[0]]


def observed_intersection(a, b, aid, bid):
    from .calibration_review import robust_fit
    ca,na,ra,sa=robust_fit(a,True)
    cb,nb,rb,sb=robust_fit(b,True)
    direction=np.cross(na,nb); norm=np.linalg.norm(direction)
    if norm<.1: raise ValueError('两个平面近乎平行')
    direction/=norm
    origin=np.linalg.lstsq(np.stack([na,nb]),[na@ca,nb@cb],rcond=None)[0]
    spacings=[]; ranges=[]
    for pts in (a,b):
        spacing=max(1e-5,float(np.median(cKDTree(pts).query(pts,k=2)[0][:,1])))
        spacings.append(spacing)
        near=np.linalg.norm(np.cross(pts-origin,direction),axis=1)<=3*spacing
        if near.sum()<3: raise ValueError('交线远离真实观测支撑')
        ranges.append(_runs((pts[near]-origin)@direction,3*spacing))
    overlaps=[(max(a0,b0),min(a1,b1)) for a0,a1 in ranges[0] for b0,b1 in ranges[1]
              if min(a1,b1)-max(a0,b0)>3*max(spacings)]
    if not overlaps: raise ValueError('两平面没有共同连续支撑的交线')
    lo,hi=max(overlaps,key=lambda v:v[1]-v[0])
    return {'endpoints':[(origin+lo*direction).tolist(),(origin+hi*direction).tolist()],
            'support':[{'indices':aid.tolist(),'residualM':float(np.median(ra)),'unsupportedIndices':aid[ra>3*sa].tolist()},
                       {'indices':bid.tolist(),'residualM':float(np.median(rb)),'unsupportedIndices':bid[rb>3*sb].tolist()}],
            'points':np.concatenate([a,b]).tolist(), 'spacingM':max(spacings),
            'label':'相邻平面有限交线（真实观测支撑，非新增回波）'}


def extract_intersections(points, progress=lambda *_:None, include_planes=False):
    from .calibration_review import robust_fit
    ids=np.flatnonzero(np.isfinite(points).all(axis=1))
    xyz=np.asarray(points[ids],float)
    n=len(xyz)
    if n<24: return [], {'finitePointCount':n,'planeCount':0,'intersectionCount':0,**({'planes':[]} if include_planes else {})}
    progress(5,'独立分析完整点云的局部平面',None)
    distance,neighbors=cKDTree(xyz).query(xyz,k=min(17,n),workers=4)
    spacing=np.maximum(np.median(distance[:,1:5],axis=1),1e-5)
    normals=np.empty((n,3)); planar=np.zeros(n,bool)
    for start in range(0,n,8192):
        end=min(n,start+8192)
        local=xyz[neighbors[start:end]]
        centered=local-local.mean(axis=1,keepdims=True)
        covariance=np.einsum('nki,nkj->nij',centered,centered)
        values,vectors=np.linalg.eigh(covariance)
        normals[start:end]=vectors[:,:,0]
        planar[start:end]=(values[:,0]<.12*np.maximum(values[:,1],1e-12)) & (values[:,1]>.02*values[:,2])
    src=np.repeat(np.arange(n),neighbors.shape[1]-1)
    dst=neighbors[:,1:].ravel()
    offset=xyz[dst]-xyz[src]
    close=np.linalg.norm(offset,axis=1)<=3*np.maximum(spacing[src],spacing[dst])
    tolerance=np.maximum(.015,.12*np.minimum(spacing[src],spacing[dst]))
    smooth=close & planar[src] & planar[dst] & (abs(np.einsum('ij,ij->i',normals[src],normals[dst]))>.97)
    smooth &= abs(np.einsum('ij,ij->i',offset,normals[src]))<=tolerance
    smooth &= abs(np.einsum('ij,ij->i',offset,normals[dst]))<=tolerance
    graph=coo_matrix((np.ones(smooth.sum(),np.uint8),(src[smooth],dst[smooth])),shape=(n,n)).tocsr()
    _,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels)
    planes={}; plane_records=[]; accepted=np.full(n,-1,int)
    order=np.argsort(labels,kind='stable'); bounds=np.r_[0,np.cumsum(counts)]
    for label in np.flatnonzero(counts>=24):
        members=order[bounds[label]:bounds[label+1]]
        try: center,normal,residual,_=robust_fit(xyz[members],True)
        except ValueError: continue
        planes[int(label)]=members;accepted[members]=label
        if include_planes:
            plane_records.append({'id':int(label),'indices':ids[members].tolist(),'center':center.tolist(),
                                  'normal':normal.tolist(),'medianResidualM':float(np.median(residual))})
    # Adjacency comes from spatial neighborhoods, not image positions.
    crossing=close & (accepted[src]>=0) & (accepted[dst]>=0) & (accepted[src]!=accepted[dst])
    pairs=np.unique(np.sort(np.column_stack([accepted[src[crossing]],accepted[dst[crossing]]]),axis=1),axis=0)
    # Normals at a crease mix both faces and are intentionally not assigned to
    # either plane. Use those boundary returns as spatial adjacency witnesses.
    boundary=np.flatnonzero(accepted<0); supported=np.flatnonzero(accepted>=0)
    if len(boundary) and len(supported)>=2:
        bd,bn=cKDTree(xyz[supported]).query(xyz[boundary],k=min(17,len(supported)),workers=4)
        nearby=supported[bn]
        left=np.repeat(accepted[nearby[:,0]],nearby.shape[1])
        right=accepted[nearby].ravel()
        reach=(bd<=4*spacing[boundary,None]).ravel() & (left!=right)
        extra=np.sort(np.column_stack([left[reach],right[reach]]),axis=1)
        pairs=np.unique(np.concatenate([pairs,extra]),axis=0)
    result=[]
    for i,(left,right) in enumerate(pairs):
        progress(20+int(35*i/max(1,len(pairs))),'计算相邻平面的有支撑有限交线',None)
        a,b=planes[int(left)],planes[int(right)]
        try: fit=observed_intersection(xyz[a],xyz[b],ids[a],ids[b])
        except ValueError: continue
        result.append({'indicesA':ids[a].tolist(),'indicesB':ids[b].tolist(),'fit':fit,
                       'planeIds':[int(left),int(right)],'extractionVersion':EXTRACTION_VERSION})
    return result, {'finitePointCount':n,'planeCount':len(planes),'intersectionCount':len(result),
                    **({'planes':plane_records} if include_planes else {})}
