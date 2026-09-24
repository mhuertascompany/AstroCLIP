"""Random observed-image/SFH examples by cluster, stellar mass and redshift."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('/Users/marchuertascompany/Documents/data/EUCLID/DR1/explorer_bundle'))
    p.add_argument('--output',type=Path,default=Path('euclid/diagnostics/random_cluster_mass_redshift'))
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=Path('euclid/diagnostics')
    cat=pd.read_csv(base/'cluster_phz_ssfr/cluster_phz_ssfr.csv',dtype={'galaxy_id':str})
    # No sSFR validity/quality selection: use only ID, cluster and mass.
    cat=cat[['galaxy_id','cluster','phz_log_stellar_mass']]
    members=pd.read_csv(base/'cluster_recent_ms_100myr/cluster_membership.csv',dtype={'galaxy_id':str})
    cat=cat.merge(members[['galaxy_id','redshift']],on='galaxy_id',validate='one_to_one')
    cat['has_stamp']=[(args.bundle/'VIS'/f'VIS_{g}.jpg').is_file() for g in cat.galaxy_id]
    mass_edges=[9.,9.5,10.,10.5,11.,11.5]
    z_edges=[0.,.3,.6,1.,1.5]
    rng=np.random.default_rng(args.seed)
    colors=['#236C9C','#B35D21','#228461']
    counts=[];selections=[];page=0
    with h5py.File(args.bundle/'euclid_explorer.h5') as h, PdfPages(args.output/'random_sfh_morphology_mass_redshift.pdf') as pdf:
        lookup={str(int(g)):i for i,g in enumerate(h['galaxy_id'][:])}
        time=h['sfh_time_grid'][:].astype(float);eps=float(h.attrs.get('sfh_log_epsilon',1e-10))
        for ml,mh in zip(mass_edges[:-1],mass_edges[1:]):
            for zl,zh in zip(z_edges[:-1],z_edges[1:]):
                subset=cat[(cat.phz_log_stellar_mass>=ml)&(cat.phz_log_stellar_mass<mh)&(cat.redshift>=zl)&(cat.redshift<zh)]
                if not len(subset):continue
                page+=1
                chosen=[]
                for k in [1,2,3]:
                    group=subset[subset.cluster==k];pool=group[group.has_stamp]
                    take=pool.loc[rng.choice(pool.index,min(3,len(pool)),replace=False)].copy()
                    take['pdf_page']=page;take['mass_bin']=f'[{ml},{mh})';take['redshift_bin']=f'[{zl},{zh})'
                    selections.append(take);chosen.append(take)
                    counts.append(dict(pdf_page=page,mass_low=ml,mass_high=mh,z_low=zl,z_high=zh,cluster=k,n_objects=len(group),n_stamps=len(pool),n_shown=len(take)))
                histories={}
                for take in chosen:
                    for g in take.galaxy_id:
                        i=lookup[g]
                        histories[g]=[np.maximum(10.**h[key][i].astype(float)-eps,0) for key in ['sfh','sfh_p16','sfh_p84']]
                ymax=max(np.max(v[2]) for v in histories.values())*1.05 if histories else 1.
                fig,axes=plt.subplots(3,6,figsize=(18,10),layout='constrained',gridspec_kw={'width_ratios':[1,1.6]*3})
                for row,take in enumerate(chosen):
                    k=row+1
                    for ax in axes[row]:ax.axis('off')
                    count=counts[-3+row]
                    for j,(_,obj) in enumerate(take.iterrows()):
                        im,ax=axes[row,2*j:2*j+2]
                        with Image.open(args.bundle/'VIS'/f'VIS_{obj.galaxy_id}.jpg') as img:im.imshow(np.asarray(img.convert('L')),cmap='gray')
                        im.set_title(f'Cluster {k} · {obj.galaxy_id}',fontsize=8,color=colors[row])
                        ax.set_axis_on();med,lo,hi=histories[obj.galaxy_id]
                        ax.fill_between(time,lo,hi,color=colors[row],alpha=.2)
                        ax.plot(time,med,color=colors[row],lw=1.3)
                        ax.axvline(100./float(h['sfh_time_norm'][lookup[obj.galaxy_id]]),color='.3',ls=':',lw=1)
                        ax.set(xlim=(0,1),ylim=(0,ymax),xlabel='Fractional lookback time',ylabel='Normalized SFH weight')
                        ax.set_title(f'log M★={obj.phz_log_stellar_mass:.2f}; z={obj.redshift:.3f}',fontsize=9)
                        ax.tick_params(labelsize=8)
                    axes[row,0].text(0,-.10,f'Cluster {k}: {count["n_objects"]:,} objects\n{count["n_stamps"]:,} stamps; {count["n_shown"]} shown',transform=axes[row,0].transAxes,fontsize=9,color=colors[row],va='top')
                    if not len(take):axes[row,1].text(.5,.5,'No available examples',ha='center',va='center',transform=axes[row,1].transAxes)
                fig.suptitle(f'{ml:g} ≤ PHZ log M★ < {mh:g}     |     {zl:g} ≤ z < {zh:g}     |     page {page}\nRandom examples in each of the three fixed SFH clusters · no sSFR selection',fontsize=14)
                fig.supxlabel('Seed 42 · up to 3 objects per cluster/bin · observed VIS stamps · common SFH y-axis within each page\nPointwise SFH uncertainty shaded; dotted line: 100 Myr. Broad bins do not guarantee matched distributions.',fontsize=10)
                pdf.savefig(fig)
                fig.savefig(args.output/f'page_{page:02d}.png',dpi=110)
                plt.close(fig)
    selected=pd.concat(selections,ignore_index=True)
    assert selected.galaxy_id.nunique()==len(selected)
    selected.to_csv(args.output/'selected_ids.csv',index=False)
    pd.DataFrame(counts).to_csv(args.output/'bin_counts.csv',index=False)
    included=(cat.phz_log_stellar_mass>=mass_edges[0])&(cat.phz_log_stellar_mass<mass_edges[-1])&(cat.redshift>=z_edges[0])&(cat.redshift<z_edges[-1])
    report=dict(seed=args.seed,mass_edges=mass_edges,redshift_edges=z_edges,n_parent=len(cat),n_in_bins=int(included.sum()),n_outside_bins=int((~included).sum()),n_selected=len(selected),pages=page,ssfr_selection=False,mass_source='PHZ catalog median stellar mass',redshift_source='explorer membership redshift',bundle=str(args.bundle))
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
