"""Three-cluster SFH/recent-activity diagnostic on a local explorer bundle."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from .recent_ms import catalog_recent_offsets, WINDOWS
from .main_sequence_sfh import main_sequence_along_sfh


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('euclid/diagnostics/cluster_recent_ms_100myr'))
    parser.add_argument('--window', choices=list(WINDOWS), default='100 Myr')
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.archive) as a:
        ids=a['galaxy_id'].astype(np.int64)
        embedding=a['sfh_embedding'].astype(float)
        xy=a['xy_sfh']
        mass=a['log_stellar_mass'].astype(float)
        z=a['redshift'].astype(float)
    with h5py.File(args.bundle/'euclid_explorer.h5') as h:
        lookup={int(g):i for i,g in enumerate(h['galaxy_id'][:])}
        rows=np.array([lookup[int(g)] for g in ids])
        sfh=h['sfh'][:][rows].astype(float)
        time=h['sfh_time_grid'][:].astype(float)
        norm=h['sfh_time_norm'][:][rows].astype(float)
        eps=float(h.attrs.get('sfh_log_epsilon',1e-10))
        lower=h['sfh_p16'][:][rows]
        upper=h['sfh_p84'][:][rows]
    finite=np.isfinite(embedding).all(axis=1)&(np.linalg.norm(embedding,axis=1)>0)
    # As in explorer dynamic clustering: KMeans on stored full embeddings.
    labels=np.full(len(ids),-1)
    labels[finite]=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(embedding[finite])
    weights=np.maximum(10**sfh-eps,0)
    weights/=weights.sum(axis=1)[:,None]
    # Stable readable numbering: increasing median mean-lookback of each group.
    order=sorted(range(3),key=lambda k:np.median((weights@time)[labels==k]))
    labels=np.array([order.index(int(k))+1 if k>=0 else -1 for k in labels])
    settings=dict(return_fraction=0.,mass_offset=0.,ms_sfr_offset=-.93,epsilon=eps)
    print('Computing recent activity for',len(ids),'objects',flush=True)
    recent=catalog_recent_offsets(sfh,time,z,mass,norm,**settings)
    delta=recent[:,list(WINDOWS).index(args.window)]
    allrows=[dict(galaxy_id=str(ids[i]),cluster=int(labels[i]),redshift=float(z[i]),
                 log_stellar_mass=float(mass[i]),delta_ms_100myr=float(recent[i,0]),
                 delta_ms_recent_01=float(recent[i,1]),delta_ms_recent_02=float(recent[i,2])) for i in range(len(ids))]
    write_csv(args.output/'cluster_membership.csv',allrows)
    for k in range(1,4):
        write_csv(args.output/f'cluster_{k}_ids.csv',[r for r in allrows if r['cluster']==k])
    colors=['#236C9C','#B35D21','#228461']
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for k,c in zip(range(1,4),colors):
        m=labels==k
        axs[0].scatter(*xy[m].T,s=2,alpha=.45,color=c,label=f'Cluster {k} (n={m.sum():,})',rasterized=True)
    axs[0].legend(markerscale=4)
    axs[0].set(xlabel='SFH UMAP 1',ylabel='SFH UMAP 2',title='KMeans in full aligned SFH space')
    valid=np.isfinite(delta)&(labels>0)
    lo=min(-4.,float(np.min(delta[valid])))
    hi=max(1.5,float(np.max(delta[valid])))
    bins=np.linspace(lo,hi,50)
    axs[1].hist(delta[valid],bins=bins,density=True,color='0.75',alpha=.5,label='Overall')
    summary=[]
    for k,c in zip(range(1,4),colors):
        m=valid&(labels==k)
        axs[1].hist(delta[m],bins=bins,density=True,histtype='step',lw=1.7,color=c,label=f'Cluster {k}')
        summary.append(dict(cluster=k,n=int((labels==k).sum()),n_valid_recent=int(m.sum()),
            median_delta=float(np.median(delta[m])),n_censored=int(np.sum(m&(delta<=-4))),
            low=int(np.sum(m&(delta<-.5))),middle=int(np.sum(m&(delta>=-.5)&(delta<=.5))),high=int(np.sum(m&(delta>.5)))))
    for x in [-.5,.5]:axs[1].axvline(x,ls=':',color='0.3')
    axs[1].legend()
    axs[1].set(xlabel=f'Recent ΔMS (last {args.window}) [dex]',ylabel='Probability density',title='Each distribution normalized independently')
    fig.suptitle('Bright validation sample · R=0 · empirical MS shift −0.93 dex\nExact zeros / values below −4 dex plotted at −4; no mass/redshift matching',fontsize=11)
    for ext in ['pdf','png']:fig.savefig(args.output/f'cluster_distributions.{ext}',dpi=180)
    plt.close(fig)
    rng=np.random.default_rng(42)
    selections=[]
    categories=[('Below MS: ΔMS < −0.5',delta<-.5),('Near MS: −0.5 ≤ ΔMS ≤ 0.5',(delta>=-.5)&(delta<=.5)),('Above MS: ΔMS > 0.5',delta>.5)]
    with PdfPages(args.output/'cluster_sfh_cutout_galleries.pdf') as pdf:
        for k in range(1,4):
            fig,axes=plt.subplots(3,8,figsize=(22,10),layout='constrained',gridspec_kw={'width_ratios':[1,1.7]*4})
            for row,(label,category) in enumerate(categories):
                candidates=np.flatnonzero((labels==k)&valid&category)
                candidates=np.array([i for i in candidates if (args.bundle/'VIS'/f'VIS_{ids[i]}.jpg').is_file()],dtype=int)
                chosen=rng.choice(candidates,min(4,len(candidates)),replace=False)
                for col in range(8):axes[row,col].set_axis_off()
                for j,i in enumerate(chosen):
                    selections.append(dict(**allrows[i],activity_range=label,selection_window=args.window,selection_delta_ms=float(delta[i])))
                    im,ax=axes[row,2*j:2*j+2]
                    im.imshow(Image.open(args.bundle/'VIS'/f'VIS_{ids[i]}.jpg'),cmap='gray')
                    im.set_title(f'{ids[i]}\nz={z[i]:.2f}; log M★={mass[i]:.2f}',fontsize=8)
                    ax.set_axis_on()
                    track=main_sequence_along_sfh(time,sfh[i],float(z[i]),float(mass[i]),time_norm_myr=float(norm[i]),**settings)
                    # Stored posterior envelopes already share preprocessing scale.
                    ax.fill_between(time,np.maximum(10.**lower[i]-eps,0),np.maximum(10.**upper[i]-eps,0),alpha=.2,color='#236C9C')
                    ax.plot(time,weights[i],color='#236C9C',lw=1.2,label='Median SFH')
                    ax.plot(time,track['weights'],color='#228461',ls='--',lw=1,label='Shifted MS')
                    mode, width = WINDOWS[args.window]
                    window_myr = width*1000 if mode == 'gyr' else width*track['observation_age_years']/1e6
                    ax.axvspan(0,min(window_myr/norm[i],1),color='0.5',alpha=.12)
                    ax.set(xlim=(0,1),ylim=(0,None),xlabel='Fractional lookback',title=f'Recent ΔMS={delta[i]:+.2f} dex')
                    ax.tick_params(labelsize=8)
                    ax.title.set_fontsize(9)
                    if j==0:ax.set_ylabel('Normalized SFH bin weight')
                    if row==0 and j==0:ax.legend(fontsize=7)
                axes[row,0].text(0, -.13,f'{label}\n{len(candidates):,} available stamps',transform=axes[row,0].transAxes,fontsize=9)
            fig.suptitle(f'Cluster {k} · last {args.window} · MS shift −0.93 dex · random examples in common recent-activity ranges\nBlue: SFH; green: shifted MS; shading: recent window / pointwise SFH uncertainty. No mass/redshift matching.',fontsize=13)
            pdf.savefig(fig)
            fig.savefig(args.output/f'cluster_{k}_gallery.png',dpi=140)
            plt.close(fig)
    write_csv(args.output/'gallery_selected_ids.csv',selections)
    (args.output/'manifest.json').write_text(json.dumps(dict(archive=str(args.archive),bundle=str(args.bundle),
        embedding='sfh_embedding',algorithm='KMeans k=3 n_init=10 seed=42 on stored embeddings',
        label_order='increasing median mean fractional lookback',window=args.window,
        settings=settings,summary=summary),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
