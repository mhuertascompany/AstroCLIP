"""Audit PHZ and median-SFH 100 Myr specific rates for matched triplets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    out=Path('euclid/diagnostics/matched_mass_ssfr_clusters')
    table=pd.read_csv(out/'selected_ids.csv',dtype={'galaxy_id':str})
    source=Path('/Users/marchuertascompany/Documents/data/EUCLID/DR1/explorer_bundle/euclid_explorer.h5')
    fractions=[];raw_fractions=[]
    with h5py.File(source) as h:
        lookup={str(int(g)):i for i,g in enumerate(h['galaxy_id'][:])}
        t=h['sfh_time_grid'][:].astype(float)
        f_edges=np.r_[0.,(t[:-1]+t[1:])/2,1.]
        eps=float(h.attrs.get('sfh_log_epsilon',1e-10))
        for g in table.galaxy_id:
            i=lookup[g]
            logs=h['sfh'][i].astype(float)
            w=np.maximum(10.**logs-eps,0.)
            w/=w.sum()
            edges=f_edges*float(h['sfh_time_norm'][i])*1e6
            overlap=np.clip(1e8-edges[:-1],0.,np.diff(edges))/np.diff(edges)
            fractions.append(float(w@overlap))
            raw=10.**logs;raw/=raw.sum()
            raw_fractions.append(float(raw@overlap))
    fraction=np.array(fractions)
    ssfr=fraction/1e8
    with np.errstate(divide='ignore'):
        log_ssfr=np.log10(ssfr)
    table['sfh_recent_mass_fraction_100myr']=fraction
    table['sfh_ssfr_100myr_per_year_R0']=ssfr
    table['sfh_log_ssfr_100myr_R0']=log_ssfr
    table['sfh_exact_zero_after_epsilon_subtraction']=fraction==0
    table['sfh_minus_phz_log_ssfr']=log_ssfr-table.phz_log_ssfr
    table['raw_fraction_without_epsilon_subtraction']=raw_fractions
    table.to_csv(out/'phz_vs_sfh_ssfr_100myr.csv',index=False)
    colors=['#236C9C','#B35D21','#228461']
    fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    x=table.phz_log_ssfr.to_numpy()
    pos=ssfr>0
    bottom=min(-13.,np.floor(log_ssfr[pos].min())-.5) if pos.any() else -13.
    ax=axs[0]
    for k,color in zip([1,2,3],colors):
        m=table.cluster.to_numpy()==k
        ax.scatter(x[m&pos],log_ssfr[m&pos],c=color,s=50)
        ax.scatter(x[m&~pos],np.full(np.sum(m&~pos),bottom),c=color,s=65,marker='v')
        for i in np.flatnonzero(m):
            ax.annotate(str(table.triplet.iloc[i]),(x[i],log_ssfr[i] if pos[i] else bottom),xytext=(4,4),textcoords='offset points',fontsize=8,color=color)
    ax.plot([-14,-8],[-14,-8],'k--',lw=1)
    ax.set(xlim=(x.min()-.2,x.max()+.2),ylim=(bottom-.3,max(-9.,log_ssfr[pos].max()+.2)),
           xlabel='PHZ log₁₀(sSFR / yr⁻¹)',ylabel='Median-SFH log₁₀(sSFR / yr⁻¹), R=0',title='Same 18 galaxies · 100 Myr in both estimates')
    handles=[Line2D([],[],color=c,marker='o',ls='',label=f'Cluster {k}') for k,c in zip([1,2,3],colors)]
    handles.append(Line2D([],[],color='k',marker='v',ls='',label='Exact zero SFH rate (not a finite log rate)'))
    ax.legend(handles=handles,fontsize=8)
    ax.grid(alpha=.15)
    ax=axs[1];ax.axis('off')
    records=[]
    for _,r in table.iterrows():
        records.append([f'{int(r.triplet)} / {int(r.cluster)}',f'{r.phz_log_ssfr:.3f}',
            'ZERO' if r.sfh_exact_zero_after_epsilon_subtraction else f'{r.sfh_log_ssfr_100myr_R0:.3f}',
            f'{r.sfh_recent_mass_fraction_100myr:.2e}'])
    cells=ax.table(cellText=records,colLabels=['Triplet / cluster','PHZ log sSFR','SFH log sSFR','Recent mass fraction'],loc='center',cellLoc='center')
    cells.auto_set_font_size(False);cells.set_fontsize(8);cells.scale(1,1.45)
    ax.set_title('Numerical audit',fontsize=12)
    for row,cluster in enumerate(table.cluster,start=1):
        cells[row,0].get_text().set_color(colors[int(cluster)-1])
    fig.supxlabel('SFH sSFR = normalized mass fraction formed in 0–100 Myr / 10⁸ yr; R=0. No MS shift.\n'
        'Uses the preprocessed pointwise median SFH with the stored log epsilon removed. Partial bins integrated uniformly.\n'
        'Numbers label triplets; zero-rate triangles are placed on a display row. This is not an audit of individual posterior draws.',fontsize=9)
    for ext in ['pdf','png']:fig.savefig(out/f'phz_vs_sfh_ssfr_100myr.{ext}',dpi=180)
    report=[]
    for k,g in table.groupby('cluster'):
        p=g.sfh_ssfr_100myr_per_year_R0>0
        report.append(dict(cluster=int(k),n=len(g),n_zero=int((~p).sum()),
            median_sfh_minus_phz_positive=float(g.loc[p,'sfh_minus_phz_log_ssfr'].median()) if p.any() else None))
    (out/'phz_vs_sfh_ssfr_100myr.json').write_text(json.dumps(dict(source=str(source),epsilon=eps,window_myr=100,return_fraction=0,summary=report),indent=2))
    print(table[['triplet','cluster','phz_log_ssfr','sfh_log_ssfr_100myr_R0','sfh_recent_mass_fraction_100myr']].to_string(index=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
