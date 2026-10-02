"""Keep the best actually evaluated finite candidate, independently of solver status."""
import numpy as np
from scipy.optimize import minimize

def solve(cost, delta_matrix, seed, optimizer=minimize):
    seed=np.array(seed,dtype=float,copy=True)
    initial=float(cost(seed))
    if not np.isfinite(initial):raise ValueError('Seed objective must be finite')
    best={'matrix':seed.copy(),'loss':initial,'source':'seed','evaluation':0}
    count=0
    cache={}
    def fun(d):
        nonlocal count
        d=np.asarray(d,dtype=float)
        if d.shape!=(6,) or not np.isfinite(d).all():return float('inf')
        key=d.tobytes()
        if key in cache:return cache[key]
        T=delta_matrix(d)@seed;value=float(cost(T));count+=1
        if not np.isfinite(value):value=float('inf')
        if value<best['loss']:
            best.update(matrix=T.copy(),loss=value,source='evaluated_candidate',evaluation=count)
        cache[key]=value
        return value
    result=optimizer(fun,np.zeros(6),method='Powell',options={'xtol':1e-5,'ftol':1e-5,'maxiter':260})
    endpoint_loss=fun(result.x)
    endpoint_d=np.asarray(result.x,dtype=float)
    endpoint=delta_matrix(endpoint_d)@seed if endpoint_d.shape==(6,) and np.isfinite(endpoint_d).all() else None
    verified=float(cost(best['matrix']))
    if not np.isclose(verified,best['loss'],rtol=1e-12,atol=1e-12):
        raise AssertionError('Objective changed when accepted candidate was reevaluated')
    if not np.isfinite(verified) or verified>initial:
        raise AssertionError('Accepted objective is nonfinite or worse than seed')
    return dict(matrix=best['matrix'].tolist(),loss=verified,initialLoss=initial,
                source=best['source'],bestEvaluation=best['evaluation'],uniqueEvaluations=count,
                endpointMatrix=None if endpoint is None else endpoint.tolist(),endpointLoss=endpoint_loss,
                optimizerSuccess=bool(result.success),optimizerMessage=str(result.message),
                optimizerIterations=int(result.nit),optimizerNfev=int(result.nfev))
