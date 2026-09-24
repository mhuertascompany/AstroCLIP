"""Compare PHZ catalog specific SFR across fixed SFH embedding clusters."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--membership',type=Path,default=Path('euclid/diagnostics/cluster_recent_ms_100myr/cluster_membership.csv'))
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('euclid/diagnostics/cluster_phz_ssfr'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.membership.open()))
    ids=np.array([int(r['galaxy_id']) for r in rows],dtype=np.int64)
    clusters=np.array([int(r['cluster']) for r in rows])
    table=Table.read(args.catalog)
    catids=np.asarray(table['object_id'],dtype=np.int64)
    if len(np.unique(catids))!=len(catids):raise ValueError('Duplicate PHZ object IDs')
    lookup={int(g):i for i,g in enumerate(catids)}
    positions=np.array([lookup.get(int(g),-1) for g in ids])
    matched=positions>=0
    safe=np.maximum(positions,0)
    matched &= np.asarray(table['physical_parameters_matched'],bool)[safe]
    def values(name):
        arr=np.asarray(np.ma.asarray(table[name],dtype=float).filled(np.nan))[safe]
        arr[~matched]=np.nan
        return arr
    sfr=values('phz_pp_median_sfr')
    mass=values('phz_pp_median_stellarmass')
    ssfr=sfr-mass
    valid=matched & np.isfinite(sfr+mass) & (clusters>0)
    with np.load(args.archive) as a:
        index={int(g):i for i,g in enumerate(a['galaxy_id'])}
        xy=a['xy_sfh'][[index[int(g)] for g in ids]]
    colors=['#236C9C','#B35D21','#228461']
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for k,c in zip(range(1,4),colors):
        m=clusters==k
        axs[0].scatter(*xy[m].T,s=2,alpha=.45,color=c,label=f'Cluster {k} (n={m.sum():,})',rasterized=True)
    axs[0].legend(markerscale=4)
    axs[0].set(xlabel='SFH UMAP 1',ylabel='SFH UMAP 2',title='Unchanged aligned-SFH clusters')
    bins=np.linspace(np.floor(ssfr[valid].min()*2)/2,np.ceil(ssfr[valid].max()*2)/2,55)
    axs[1].hist(ssfr[valid],bins=bins,density=True,color='0.75',alpha=.5,label='Overall')
    summary=[]
    for k,c in zip(range(1,4),colors):
        m=valid&(clusters==k)
        median=float(np.median(ssfr[m]))
        axs[1].hist(ssfr[m],bins=bins,density=True,histtype='step',lw=1.7,color=c,label=f'Cluster {k}: median {median:.2f}')
        summary.append(dict(cluster=k,n=int((clusters==k).sum()),n_valid=int(m.sum()),median_log_ssfr=median,p16_p84=np.percentile(ssfr[m],[16,84]).tolist()))
    axs[1].legend(fontsize=9)
    axs[1].set(xlabel='PHZ log₁₀(sSFR / yr⁻¹)',ylabel='Probability density',title='Each distribution normalized independently')
    fig.suptitle('PHZ 100 Myr SFR / PHZ stellar mass · no empirical MS shift\nRatio of catalog medians; no mass/redshift matching or PHZ quality cuts',fontsize=11)
    for ext in ['pdf','png']:fig.savefig(args.output/f'cluster_distributions.{ext}',dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    # Shuffle draw order so the last cluster does not systematically hide others.
    shown = np.flatnonzero(valid)
    shown = np.random.default_rng(42).permutation(shown)
    point_colors = np.asarray(colors)[clusters[shown]-1]
    ax.scatter(mass[shown], ssfr[shown], c=point_colors, s=7, alpha=.4,
               linewidths=0, rasterized=True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', color=c,
               label=f'Cluster {k} (n={np.sum(valid & (clusters==k)):,})')
               for k,c in zip(range(1,4), colors)]
    ax.legend(handles=handles, loc='lower left')
    ax.set(xlabel='PHZ log₁₀(M★ / M☉)', ylabel='PHZ log₁₀(sSFR / yr⁻¹)',
           title='Catalog sSFR versus stellar mass · fixed SFH clusters')
    ax.grid(alpha=.15)
    fig.supxlabel('PHZ 100 Myr SFR / stellar mass · ratio of catalog medians\n'
                  'All finite matched objects; no MS shift or PHZ quality cuts.', fontsize=10)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'ssfr_vs_mass_clusters.{ext}', dpi=200)
    plt.close(fig)
    with (args.output/'cluster_phz_ssfr.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['galaxy_id','cluster','phz_log_sfr','phz_log_stellar_mass','phz_log_ssfr','valid'])
        writer.writerows((str(ids[i]),int(clusters[i]),sfr[i],mass[i],ssfr[i],bool(valid[i])) for i in range(len(ids)))
    report=dict(catalog=str(args.catalog),membership=str(args.membership),formula='phz_pp_median_sfr - phz_pp_median_stellarmass',n=int(len(ids)),n_valid=int(valid.sum()),summary=summary)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
