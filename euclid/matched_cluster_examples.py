"""Observed stamps/SFH triplets with closely matched PHZ sSFR, mass and z."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE=Path('euclid/diagnostics')
BUNDLE=Path('/Users/marchuertascompany/Documents/data/EUCLID/DR1/explorer_bundle')
OUT=BASE/'matched_mass_ssfr_clusters'
COLORS=['#236C9C','#B35D21','#228461']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cat=pd.read_csv(BASE/'cluster_phz_ssfr/cluster_phz_ssfr.csv',dtype={'galaxy_id':str})
    members=pd.read_csv(BASE/'cluster_recent_ms_100myr/cluster_membership.csv',dtype={'galaxy_id':str})
    cat=cat.merge(members[['galaxy_id','redshift']],on='galaxy_id',validate='one_to_one')
    cat=cat[cat.phz_log_stellar_mass.between(10,10.5)&np.isfinite(cat.phz_log_ssfr+cat.redshift)].copy()
    cat=cat[[ (BUNDLE/'VIS'/f'VIS_{g}.jpg').is_file() for g in cat.galaxy_id]].reset_index(drop=True)
    used=set();groups=[];report=[]
    # Fixed targets chosen in the shared support; deterministic matching.
    for target in [-10.2,-9.9,-9.6]:
        for repeat in range(2):
            pools=[cat[(cat.cluster==k)&(abs(cat.phz_log_ssfr-target)<=.12)&~cat.galaxy_id.isin(used)] for k in [1,2,3]]
            candidates=[]
            for _,anchor in pools[2].iterrows():
                options=[]
                for pool in pools[:2]:
                    distance=((pool.phz_log_ssfr-anchor.phz_log_ssfr)/.08)**2+((pool.phz_log_stellar_mass-anchor.phz_log_stellar_mass)/.1)**2+((pool.redshift-anchor.redshift)/.1)**2
                    options.append(pool.loc[distance.nsmallest(8).index])
                for _,a in options[0].iterrows():
                    for _,b in options[1].iterrows():
                        triple=pd.DataFrame([a,b,anchor])
                        spreads=np.array([np.ptp(triple.phz_log_ssfr),np.ptp(triple.phz_log_stellar_mass),np.ptp(triple.redshift)])
                        if np.all(spreads<=np.array([.1,.15,.15])):
                            score=np.sum((spreads/np.array([.1,.15,.15]))**2)+.1*abs(triple.phz_log_ssfr.mean()-target)/.12
                            candidates.append((score,triple,spreads))
            if not candidates:
                report.append(dict(target=target,status='No unused triple meets tolerances'))
                continue
            _,triple,spreads=min(candidates,key=lambda x:x[0])
            number=len(groups)+1
            triple['triplet']=number;triple['target_log_ssfr']=target
            groups.append(triple);used.update(triple.galaxy_id)
            report.append(dict(triplet=number,target=target,ssfr_spread=float(spreads[0]),mass_spread=float(spreads[1]),redshift_spread=float(spreads[2])))
    if not groups:raise ValueError('No matched groups')
    selected=pd.concat(groups,ignore_index=True)
    selected.to_csv(OUT/'selected_ids.csv',index=False)
    with h5py.File(BUNDLE/'euclid_explorer.h5') as h:
        lookup={str(int(g)):i for i,g in enumerate(h['galaxy_id'][:])}
        time=h['sfh_time_grid'][:]
        eps=float(h.attrs.get('sfh_log_epsilon',1e-10))
        histories={}
        recent_x={}
        for g in selected.galaxy_id:
            i=lookup[g]
            recent_x[g]=100./float(h['sfh_time_norm'][i])
            histories[g]=[np.maximum(10.**h[key][i].astype(float)-eps,0) for key in ['sfh','sfh_p16','sfh_p84']]
    with PdfPages(OUT/'matched_sfh_morphology.pdf') as pdf:
        for start in range(0,len(groups),3):
            batch=groups[start:start+3]
            fig,axs=plt.subplots(len(batch),6,figsize=(18,3.3*len(batch)),squeeze=False,layout='constrained',gridspec_kw={'width_ratios':[1,1.6]*3})
            for row,triple in enumerate(batch):
                ymax=max(np.max(histories[g][2]) for g in triple.galaxy_id)*1.05
                for j,(_,obj) in enumerate(triple.iterrows()):
                    im,ax=axs[row,2*j:2*j+2]
                    with Image.open(BUNDLE/'VIS'/f'VIS_{obj.galaxy_id}.jpg') as stamp:im.imshow(np.asarray(stamp.convert('L')),cmap='gray')
                    im.axis('off')
                    im.set_title(f'Cluster {obj.cluster} · {obj.galaxy_id}',fontsize=8,color=COLORS[j])
                    med,lo,hi=histories[obj.galaxy_id]
                    ax.fill_between(time,lo,hi,color=COLORS[j],alpha=.18)
                    ax.plot(time,med,color=COLORS[j],lw=1.3)
                    ax.axvline(recent_x[obj.galaxy_id],color='0.25',ls=':',lw=1.2)
                    ax.set(xlim=(0,1),ylim=(0,ymax),xlabel='Fractional lookback time',ylabel='Normalized SFH weight')
                    ax.set_title(f'Triplet {int(obj.triplet)} · log sSFR={obj.phz_log_ssfr:.3f}\nlog M★={obj.phz_log_stellar_mass:.3f}; z={obj.redshift:.3f}',fontsize=9)
                    ax.tick_params(labelsize=8)
            fig.suptitle('Different SFH clusters at matched PHZ sSFR and mass\n10 ≤ log M★ ≤ 10.5; within-triplet spans: Δlog sSFR ≤ 0.10, Δlog M★ ≤ 0.15, Δz ≤ 0.15',fontsize=12)
            fig.supxlabel('Observed VIS stamps · PHZ 100 Myr SFR / PHZ stellar mass, no MS correction · same SFH y-scale within each triplet\nDotted vertical line: 100 Myr before observation. Shading: pointwise SFH uncertainty. Examples selected for close matching.',fontsize=9)
            pdf.savefig(fig);fig.savefig(OUT/f'gallery_page_{start//3+1}.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for ax in axs.flat:ax.set_visible(False)
        for ax,triple in zip(axs.flat,groups):
            ax.set_visible(True)
            for j,(_,obj) in enumerate(triple.iterrows()):
                ax.plot(time,histories[obj.galaxy_id][0],color=COLORS[j],label=f'Cluster {obj.cluster}')
                ax.axvline(recent_x[obj.galaxy_id],color=COLORS[j],ls=':',lw=1,alpha=.8)
            ax.set(xlim=(0,1),ylim=(0,None),xlabel='Fractional lookback time',ylabel='Normalized SFH weight',title=f'Triplet {int(triple.triplet.iloc[0])} · log sSFR ≈ {triple.phz_log_ssfr.mean():.2f}')
        axs.flat[0].legend(fontsize=8)
        fig.suptitle('Direct SFH comparison within matched triplets (median histories)\nDotted lines: 100 Myr before observation, one per galaxy',fontsize=13)
        pdf.savefig(fig);fig.savefig(OUT/'sfh_overlays.png',dpi=160);plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps(dict(mass_range=[10,10.5],matching='Deterministic nearest triplets; no reuse; match PHZ sSFR, PHZ mass, redshift',tolerances=dict(log_ssfr=.1,log_mass=.15,redshift=.15),groups=report,n_selected=len(selected)),indent=2))
    assert selected.galaxy_id.nunique()==len(selected)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
