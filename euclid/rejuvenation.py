"""Exploratory active–quiet–active SFH diagnostics in fractional time."""
import numpy as np

DEFAULTS = dict(recent_width=.05, lull_width=.05, contrast=5.,
                min_old_fraction=.5, min_recent_fraction=.01)
LABELS = {
    'candidate': 'Rejuvenation: median-SFH candidate (0/1)',
    'recent_fraction': 'Rejuvenation: recent mass fraction',
    'lull_start': 'Rejuvenation: lull start (fractional lookback)',
    'log_recovery': 'Rejuvenation: log10 recent/lull rate (cap 6)',
    'old_fraction': 'Rejuvenation: mass fraction older than lull',
    'posterior_fraction': 'Rejuvenation: fraction of valid posterior draws',
    'valid_draws': 'Rejuvenation: valid posterior draws',
}


def diagnose(log_sfh, time, epsilon=1e-10, **settings):
    p = dict(DEFAULTS, **settings)
    rw, lw = p['recent_width'], p['lull_width']
    if not (0 < rw < 1 and 0 < lw and rw+2*lw <= 1 and p['contrast'] > 1
            and 0 <= p['min_old_fraction'] <= 1 and 0 <= p['min_recent_fraction'] <= 1):
        raise ValueError('Invalid rejuvenation thresholds.')
    time = np.asarray(time, float)
    sfh = np.asarray(log_sfh, float)
    if (sfh.ndim != 2 or time.ndim != 1 or sfh.shape[1] != len(time)
            or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0)
            or time[0]<0 or time[-1]>1):
        raise ValueError('Expected SFH rows on increasing fractional grid.')
    edges = np.r_[0., (time[1:]+time[:-1])/2, 1.]
    widths = np.diff(edges)
    w = np.maximum(10.**sfh-epsilon, 0.)
    total = w.sum(axis=1)
    valid = np.isfinite(w).all(axis=1) & (total>epsilon)
    w = np.divide(w,total[:,None],out=np.zeros_like(w),where=valid[:,None])
    def mass(a,b):
        return w @ (np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))/widths)
    recent = mass(0,rw)
    n=len(w)
    result={k:np.full(n,np.nan) for k in ['candidate','recent_fraction','lull_start','log_recovery','old_fraction']}
    result['candidate'][valid]=0
    result['recent_fraction'][valid]=recent[valid]
    best=np.full(n,-np.inf)
    # A fixed-width lull, scanned at native edges. Older activity must exist
    # in the immediately preceding window, not merely somewhere in the past.
    starts=np.unique(np.r_[rw,edges[(edges>=rw)&(edges<=1-2*lw)]])
    for start in starts:
        if start+2*lw>1+1e-12: continue
        lull=mass(start,start+lw)/lw
        old_rate=mass(start+lw,start+2*lw)/lw
        old_mass=mass(start+lw,1.)
        recent_rate=recent/rw
        score=np.minimum(recent_rate,old_rate)/np.maximum(lull,1e-6)
        eligible=(valid & (recent>=p['min_recent_fraction']) & (old_mass>=p['min_old_fraction']))
        better=eligible & (score>best)
        best[better]=score[better]
        result['lull_start'][better]=start
        result['old_fraction'][better]=old_mass[better]
        result['log_recovery'][better]=np.log10(np.clip(recent_rate[better]/np.maximum(lull[better],1e-6),1e-6,1e6))
    result['candidate'][valid]=(best[valid]>=p['contrast']).astype(float)
    return result


def catalog_diagnostics(source, rows, settings=None):
    """Read in bounded batches; posterior score only when draws are present."""
    rows=np.asarray(rows,int)
    time=source['sfh_time_grid'][:]
    epsilon=float(source.attrs.get('sfh_log_epsilon',1e-10))
    settings=DEFAULTS if settings is None else settings
    parts={}
    for start in range(0,len(rows),64):
        batch=rows[start:start+64]; order=np.argsort(batch); inverse=np.argsort(order)
        def read(name): return np.asarray(source[name][batch[order]])[inverse]
        values=diagnose(read('sfh'),time,epsilon,**settings)
        if 'sfh_realizations' in source:
            draws=read('sfh_realizations')
            flags=diagnose(draws.reshape(-1,len(time)),time,epsilon,**settings)['candidate'].reshape(draws.shape[:2])
            if 'sfh_realization_valid' in source:
                flags[~read('sfh_realization_valid').astype(bool)]=np.nan
            count=np.isfinite(flags).sum(axis=1)
            values['valid_draws']=count.astype(float)
            values['posterior_fraction']=np.divide(np.nansum(flags,axis=1),count,out=np.full(len(count),np.nan),where=count>0)
        for key,value in values.items(): parts.setdefault(key,[]).append(value)
    return {key:np.concatenate(value) for key,value in parts.items()}
