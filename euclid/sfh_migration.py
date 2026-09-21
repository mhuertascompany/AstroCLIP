"""SFH migration diagnostics inspired by Arango-Toro et al. (2025).

Uses normalized formed mass (constant recycling cancels) and equal-width
SFR averaging windows; this is not an exact reproduction of CIGALE outputs.
"""
import numpy as np


def migration(log_sfh, time, lag=.1, window=.02, epsilon=1e-10):
    time=np.asarray(time,float);sfh=np.asarray(log_sfh,float)
    if (sfh.ndim!=2 or time.ndim!=1 or sfh.shape[1]!=len(time) or len(time)<2
            or not np.isfinite(time).all() or np.any(np.diff(time)<=0)
            or time[0]<0 or time[-1]>1):
        raise ValueError('Expected SFH rows and increasing fractional grid.')
    n=len(sfh)
    lag=np.broadcast_to(np.asarray(lag,float),(n,))
    window=np.broadcast_to(np.asarray(window,float),(n,))
    edges=np.r_[0.,(time[1:]+time[:-1])/2,1.]
    widths=np.diff(edges)
    w=np.maximum(10.**sfh-epsilon,0.)
    total=w.sum(axis=1)
    valid=np.isfinite(w).all(axis=1)&(total>epsilon)
    w=np.divide(w,total[:,None],out=np.zeros_like(w),where=valid[:,None])
    def mass(a,b):
        a=np.broadcast_to(a,(n,));b=np.broadcast_to(b,(n,))
        overlap=np.maximum(0,np.minimum(edges[None,1:],b[:,None])-np.maximum(edges[None,:-1],a[:,None]))
        return np.sum(w*overlap/widths,axis=1)
    recent=mass(0,window);past=mass(lag,lag+window)
    older=mass(lag,1.)
    valid &= np.isfinite(lag)&np.isfinite(window)&(lag>0)&(window>0)&(window<=lag)&(lag+window<=1)
    measurable=valid&(recent>epsilon)&(past>epsilon)&(older>epsilon)
    dm=np.full(n,np.nan);ds=dm.copy()
    dm[measurable]=-np.log10(older[measurable])
    ds[measurable]=np.log10(recent[measurable]/past[measurable])
    distance=np.hypot(dm,ds)
    angle=np.degrees(np.arctan2(ds,dm));angle[distance==0]=np.nan
    return dict(angle=angle, displacement=distance, speed_fraction=distance/lag,
                delta_log_sfr=ds,delta_log_mass=dm,valid=measurable.astype(float))


def formation_times(log_sfh,time,epsilon=1e-10):
    time=np.asarray(time,float);sfh=np.asarray(log_sfh,float)
    edges=np.r_[0.,(time[1:]+time[:-1])/2,1.]
    w=np.maximum(10.**sfh-epsilon,0.);totals=w.sum(axis=1)
    valid=np.isfinite(w).all(axis=1)&(totals>epsilon)
    w=np.divide(w,totals[:,None],out=np.zeros_like(w),where=valid[:,None])
    cumulative=np.cumsum(w,axis=1)
    result={}
    for percent in (50,90):
        # At formation percentile p, fraction 1-p formed more recently.
        target=1-percent/100
        index=np.argmax(cumulative>=target,axis=1)
        before=np.where(index>0,cumulative[np.arange(len(w)),np.maximum(index-1,0)],0)
        amount=w[np.arange(len(w)),index]
        fraction=np.divide(target-before,amount,out=np.zeros(len(w)),where=amount>0)
        value=edges[index]+fraction*np.diff(edges)[index]
        value[~valid]=np.nan
        result[f'T{percent}']=value
    return result


def catalog_migration(source,rows,mode='Fractional time',lag=.1,window=.02):
    if not (np.isfinite(lag) and np.isfinite(window) and lag>0 and 0<window<=lag):
        raise ValueError('Require 0 < averaging window <= lag.')
    rows=np.asarray(rows,int);time=source['sfh_time_grid'][:]
    epsilon=float(source.attrs.get('sfh_log_epsilon',1e-10));parts={}
    if mode=='Gyr' and 'sfh_time_norm' not in source:
        raise ValueError('Gyr mode requires sfh_time_norm in the HDF5.')
    for start in range(0,len(rows),128):
        batch=rows[start:start+128];order=np.argsort(batch);inverse=np.argsort(order)
        def read(key):return np.asarray(source[key][batch[order]])[inverse]
        sfh=read('sfh')
        age=read('sfh_time_norm')/1000 if 'sfh_time_norm' in source else np.full(len(batch),np.nan)
        safe_age=np.where(np.isfinite(age)&(age>0),age,np.nan)
        fractional_lag=lag/safe_age if mode=='Gyr' else lag
        fractional_window=window/safe_age if mode=='Gyr' else window
        values=migration(sfh,time,fractional_lag,fractional_window,epsilon)
        values['speed_gyr']=values['displacement']/(lag if mode=='Gyr' else lag*safe_age)
        values.update(formation_times(sfh,time,epsilon))
        for key in ('T50','T90'):values[key+'_gyr']=values[key]*safe_age
        for key,value in values.items():parts.setdefault(key,[]).append(value)
    labels={'angle':'angle Φ (deg)', 'displacement':'displacement (dex)',
            'speed_fraction':'speed (dex / fractional time)', 'speed_gyr':'speed (dex/Gyr)',
            'delta_log_sfr':'Δlog SFR (recent − past)', 'delta_log_mass':'Δlog formed mass',
            'valid':'measurable endpoints (0/1)', 'T50':'T50 formed (fractional lookback)',
            'T90':'T90 formed (fractional lookback)', 'T50_gyr':'T50 formed (Gyr)',
            'T90_gyr':'T90 formed (Gyr)'}
    tag=f'Δ={lag:g}, avg={window:g} '+('Gyr' if mode=='Gyr' else 'fraction')
    return {f'Migration [{tag}]: {labels[key]}':np.concatenate(value) for key,value in parts.items()}
