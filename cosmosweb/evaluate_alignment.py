"""
Alignment evaluation for COSMOS-Web ZooBOT CLIP.

For each physical / morphological property we compute:
  - k-NN cross-modal prediction   (img->sfh or sfh->img)
  - k-NN in-modality baseline     (sfh->sfh or img->img, excluding self)
  - Linear probe cross-modal      (Ridge, 5-fold CV)
  - Linear probe in-modality
  - MLP probe cross-modal         (2-layer, 5-fold CV)
  - MLP probe in-modality

All results saved to a multi-page PDF.

Usage (from repo root):
    python -m cosmosweb.evaluate_alignment \
        --npz    /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_umap_zoobot_v1.npz \
        --h5     /n03data/huertas/COSMOS-Web/cosmosweb_clip/cosmosweb_dataset_v2.h5 \
        --output /n03data/huertas/COSMOS-Web/cosmosweb_clip/alignment_eval_zoobot_v1.pdf \
        --n_eval 0 \
        --n_probe 30000 \
        --k 10
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

SFH_N_BINS = 50
SFH_EPS    = 1e-10
T_FRAC     = np.linspace(0, 1, SFH_N_BINS)


# ── helpers ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20:
        return dict(pearson_r=np.nan, r2=np.nan)
    yt, yp = y_true[mask], y_pred[mask]
    return dict(pearson_r=float(stats.pearsonr(yt, yp)[0]),
                r2=float(r2_score(yt, yp)))


def knn_topk_chunked(query_emb, gallery_emb, k, chunk=1000, exclude_self=False):
    """
    Top-k gallery indices for every query, computed in chunks.
    If exclude_self=True, assumes query_emb is gallery_emb (same-modality)
    and masks the diagonal so a galaxy is not its own neighbour.
    Returns int32 array (N, k).
    """
    N    = len(query_emb)
    topk = np.empty((N, k), dtype=np.int32)
    for s in range(0, N, chunk):
        e   = min(s + chunk, N)
        sim = query_emb[s:e] @ gallery_emb.T          # (chunk, N)
        if exclude_self:
            sim[np.arange(e - s), np.arange(s, e)] = -np.inf
        topk[s:e] = np.argpartition(sim, -k, axis=1)[:, -k:].astype(np.int32)
    return topk


def knn_predict_from_topk(topk_indices, gallery_vals):
    return np.nanmean(gallery_vals[topk_indices], axis=1)


def recall_at_ks_chunked(query_emb, gallery_emb, ks, chunk=1000):
    N    = len(query_emb)
    hits = {k: 0 for k in ks}
    for s in range(0, N, chunk):
        e      = min(s + chunk, N)
        sim    = query_emb[s:e] @ gallery_emb.T
        labels = np.arange(s, e)
        for k in ks:
            top_k  = np.argpartition(sim, -k, axis=1)[:, -k:]
            hits[k] += (top_k == labels[:, None]).any(axis=1).sum()
    return [hits[k] / N for k in ks]


def rank_percentile_chunked(query_emb, gallery_emb, chunk=1000):
    N     = len(query_emb)
    ranks = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        e    = min(s + chunk, N)
        sim  = query_emb[s:e] @ gallery_emb.T
        diag = sim[np.arange(e - s), np.arange(s, e)]
        ranks[s:e] = (sim > diag[:, None]).mean(axis=1)
    return ranks


def matched_vs_random_sim(query_emb, gallery_emb, rng, chunk=1000):
    N         = len(query_emb)
    rand_perm = rng.permutation(N)
    matched   = np.empty(N, dtype=np.float32)
    random    = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = query_emb[s:e] @ gallery_emb.T
        matched[s:e] = sim[np.arange(e - s), np.arange(s, e)]
        random[s:e]  = sim[np.arange(e - s), rand_perm[s:e]]
    return matched, random


def linear_probe(X, y, cv=5, alpha=1.0):
    mask = np.isfinite(y)
    if mask.sum() < 50:
        return np.nan
    Xs = StandardScaler().fit_transform(X[mask])
    return float(cross_val_score(Ridge(alpha=alpha), Xs, y[mask],
                                 cv=cv, scoring='r2', n_jobs=-1).mean())


def mlp_probe(X, y, cv=5):
    """2-layer MLP probe with early stopping; returns cross-val R²."""
    mask = np.isfinite(y)
    if mask.sum() < 50:
        return np.nan
    Xs  = StandardScaler().fit_transform(X[mask])
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu',
                       max_iter=300, early_stopping=True, n_iter_no_change=15,
                       random_state=42)
    return float(cross_val_score(mlp, Xs, y[mask],
                                 cv=cv, scoring='r2', n_jobs=-1).mean())


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',        type=Path, required=True)
    p.add_argument('--h5',         type=Path, required=True)
    p.add_argument('--output',     type=Path,
                   default=Path('alignment_eval_zoobot_v1.pdf'))
    p.add_argument('--n_eval',     type=int, default=0,
                   help='Subsample for retrieval metrics (0 = all)')
    p.add_argument('--n_probe',    type=int, default=30000,
                   help='Subsample for linear/MLP probes (0 = same as n_eval)')
    p.add_argument('--k',          type=int, default=10)
    p.add_argument('--batch_eval', type=int, default=128)
    p.add_argument('--n_batches',  type=int, default=200)
    p.add_argument('--chunk',      type=int, default=1000)
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng  = np.random.default_rng(42)

    # 1. load embeddings
    log.info('Loading npz: %s', args.npz)
    npz = np.load(args.npz, allow_pickle=True)
    img_emb_full = npz['img_emb'].astype(np.float32)
    sfh_emb_full = npz['sfh_emb'].astype(np.float32)
    N_FULL, D    = img_emb_full.shape
    log.info('  Full dataset: %d galaxies  dim=%d', N_FULL, D)

    # subsample for retrieval metrics
    if args.n_eval > 0 and args.n_eval < N_FULL:
        eval_idx = rng.choice(N_FULL, args.n_eval, replace=False)
        eval_idx.sort()
    else:
        eval_idx = np.arange(N_FULL)
    N = len(eval_idx)
    log.info('  Retrieval eval on %d galaxies', N)

    img_emb = img_emb_full[eval_idx]
    sfh_emb = sfh_emb_full[eval_idx]
    del img_emb_full, sfh_emb_full

    # 2. catalogue properties
    props = {}
    for key, label in [
        ('zfinal',               'Redshift z'),
        ('sersic',               'Sersic n'),
        ('radius_sersic',        'Sersic radius'),
        ('axratio_sersic',       'Axis ratio b/a'),
        ('log_mass',             'log M*'),
        ('log_sfr',              'log SFR'),
        ('log_ssfr',             'log sSFR'),
        ('age_form',             'Formation age [Myr]'),
        ('log_sfr_inst',         'log SFR_inst'),
        ('log_sfr_100myr',       'log SFR_100Myr'),
        ('sfr_mass_vector_dir',  'SFR-M* dir.'),
        ('sfr_mass_vector_norm', 'SFR-M* norm'),
        ('family_elliptical',    'P(Elliptical)'),
        ('family_s0',            'P(S0)'),
        ('family_early_disk',    'P(Early disk)'),
        ('family_late_disk',     'P(Late disk)'),
    ]:
        if key in npz:
            props[label] = npz[key].astype(np.float32)[eval_idx]

    h5_indices = npz['h5_indices'].astype(int)[eval_idx]
    log.info('Loading SFH vectors: %s', args.h5)
    with h5py.File(args.h5, 'r') as f:
        sfh_log = f['sfh'][list(h5_indices)].astype(np.float32)

    sfr  = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr /= np.where(sfr.sum(1, keepdims=True) > 0,
                    sfr.sum(1, keepdims=True), 1.0)
    t    = T_FRAC[np.newaxis, :].astype(np.float32)
    props['SFH: log(old/recent)']      = sfh_log[:, -1] - sfh_log[:, 0]
    props['SFH: mean formation epoch'] = (sfr * t).sum(axis=1)
    props['SFH: f(recent 20%)']        = sfr[:, :SFH_N_BINS // 5].sum(axis=1)
    props['SFH: peak lookback t_frac'] = T_FRAC[np.argmax(sfr, axis=1)].astype(np.float32)
    del sfh_log, sfr, t

    MORPH_PROPS = [p for p in ['Sersic n', 'Sersic radius', 'Axis ratio b/a',
                   'P(Elliptical)', 'P(S0)', 'P(Early disk)', 'P(Late disk)']
                   if p in props]
    SFH_PROPS   = [p for p in ['log M*', 'log SFR', 'log sSFR',
                   'Formation age [Myr]', 'log SFR_inst', 'log SFR_100Myr',
                   'SFR-M* dir.', 'SFR-M* norm',
                   'SFH: log(old/recent)', 'SFH: mean formation epoch',
                   'SFH: f(recent 20%)', 'SFH: peak lookback t_frac']
                   if p in props]
    log.info('Properties: %s', list(props.keys()))

    # 3. alignment diagnostics (all N)
    log.info('Matched vs random cosine similarity...')
    matched_sim, random_sim = matched_vs_random_sim(img_emb, sfh_emb, rng, args.chunk)
    t_stat, p_val = stats.ttest_ind(matched_sim, random_sim)
    log.info('  matched mean=%.4f std=%.4f  random mean=%.4f std=%.4f  t=%.1f p=%.2e',
             matched_sim.mean(), matched_sim.std(),
             random_sim.mean(),  random_sim.std(), t_stat, p_val)

    log.info('Rank percentiles...')
    rank_pctile = rank_percentile_chunked(img_emb, sfh_emb, args.chunk)
    pct_top1  = (rank_pctile < 0.01).mean() * 100
    pct_top5  = (rank_pctile < 0.05).mean() * 100
    pct_top10 = (rank_pctile < 0.10).mean() * 100
    log.info('  median=%.4f  top1%%=%.1f%%  top5%%=%.1f%%  top10%%=%.1f%%',
             np.median(rank_pctile), pct_top1, pct_top5, pct_top10)

    log.info('Recall@k...')
    ks      = [k for k in [1, 5, 10, 50, 100, 500] if k <= N]
    rec_i2s = recall_at_ks_chunked(img_emb, sfh_emb, ks, args.chunk)
    rec_s2i = recall_at_ks_chunked(sfh_emb, img_emb, ks, args.chunk)
    recall_df = pd.DataFrame({'k': ks, 'img->sfh': rec_i2s, 'sfh->img': rec_s2i,
                              'random': [k / N for k in ks]}).set_index('k')
    log.info('\n%s', recall_df.to_string(float_format='{:.4f}'.format))

    log.info('Within-batch rank-1 (batch=%d)...', args.batch_eval)
    r1_i2s, r1_s2i = [], []
    for _ in range(args.n_batches):
        idx = rng.choice(N, args.batch_eval, replace=False)
        ie, se = img_emb[idx], sfh_emb[idx]
        r1_i2s.append(((ie @ se.T).argmax(1) == np.arange(args.batch_eval)).mean())
        r1_s2i.append(((se @ ie.T).argmax(1) == np.arange(args.batch_eval)).mean())
    log.info('  img->sfh=%.3f  sfh->img=%.3f  random=%.3f',
             np.mean(r1_i2s), np.mean(r1_s2i), 1 / args.batch_eval)

    # 4. k-NN property prediction — cross-modal and in-modality
    # One similarity pass per direction (cross-modal) + one per modality (in-modality).
    log.info('Computing top-k indices (4 passes)...')
    topk_i2s  = knn_topk_chunked(img_emb, sfh_emb, args.k, args.chunk)               # img -> sfh space
    topk_s2i  = knn_topk_chunked(sfh_emb, img_emb, args.k, args.chunk)               # sfh -> img space
    topk_img  = knn_topk_chunked(img_emb, img_emb, args.k, args.chunk, exclude_self=True)  # img -> img
    topk_sfh  = knn_topk_chunked(sfh_emb, sfh_emb, args.k, args.chunk, exclude_self=True)  # sfh -> sfh

    log.info('Predicting properties...')
    knn_cross_sfh   = {lbl: knn_predict_from_topk(topk_i2s, props[lbl]) for lbl in SFH_PROPS}
    knn_cross_morph = {lbl: knn_predict_from_topk(topk_s2i, props[lbl]) for lbl in MORPH_PROPS}
    knn_self_sfh    = {lbl: knn_predict_from_topk(topk_sfh, props[lbl]) for lbl in SFH_PROPS}
    knn_self_morph  = {lbl: knn_predict_from_topk(topk_img, props[lbl]) for lbl in MORPH_PROPS}

    def build_knn_df(cross_dict, self_dict, prop_list):
        rows = []
        for lbl in prop_list:
            cm = regression_metrics(props[lbl], cross_dict[lbl])
            sm = regression_metrics(props[lbl], self_dict[lbl])
            rows.append({'property': lbl,
                         'r_cross': cm['pearson_r'], 'r2_cross': cm['r2'],
                         'r_self':  sm['pearson_r'], 'r2_self':  sm['r2']})
        return pd.DataFrame(rows).set_index('property')

    df_knn_sfh   = build_knn_df(knn_cross_sfh,   knn_self_sfh,   SFH_PROPS)
    df_knn_morph = build_knn_df(knn_cross_morph, knn_self_morph, MORPH_PROPS)

    log.info('\n-- k-NN SFH props (img->sfh cross  |  sfh->sfh self) --\n%s',
             df_knn_sfh.to_string(float_format='{:.3f}'.format))
    log.info('\n-- k-NN Morph props (sfh->img cross  |  img->img self) --\n%s',
             df_knn_morph.to_string(float_format='{:.3f}'.format))

    # 5. Linear + MLP probes  (on n_probe subsample to keep runtime sane)
    n_probe = args.n_probe if (args.n_probe > 0 and args.n_probe < N) else N
    if n_probe < N:
        probe_idx = rng.choice(N, n_probe, replace=False)
        probe_idx.sort()
        img_p = img_emb[probe_idx]
        sfh_p = sfh_emb[probe_idx]
        props_p = {k: v[probe_idx] for k, v in props.items()}
    else:
        img_p, sfh_p, props_p = img_emb, sfh_emb, props
    log.info('Probes on %d galaxies (linear + MLP)...', n_probe)

    def run_probes(X_cross, X_self, prop_list, label):
        rows = []
        for lbl in prop_list:
            y = props_p[lbl]
            log.info('  %s: %s', label, lbl)
            rows.append({'property': lbl,
                         'lin_cross': linear_probe(X_cross, y),
                         'lin_self':  linear_probe(X_self,  y),
                         'mlp_cross': mlp_probe(X_cross,    y),
                         'mlp_self':  mlp_probe(X_self,     y)})
        return pd.DataFrame(rows).set_index('property')

    df_probe_sfh   = run_probes(img_p, sfh_p, SFH_PROPS,   'SFH props')
    df_probe_morph = run_probes(sfh_p, img_p, MORPH_PROPS, 'Morph props')

    log.info('\n-- Probes SFH props --\n%s',
             df_probe_sfh.to_string(float_format='{:.3f}'.format))
    log.info('\n-- Probes Morph props --\n%s',
             df_probe_morph.to_string(float_format='{:.3f}'.format))

    # 6. morphology AUC-ROC (cross-modal + in-modality, linear + MLP)
    log.info('Morphology AUC-ROC...')
    class_rows = []
    for prop, thresh in [('P(Elliptical)', 0.5), ('P(S0)', 0.5),
                         ('P(Early disk)', 0.5), ('P(Late disk)', 0.5)]:
        if prop not in props_p:
            continue
        y_full = props_p[prop]
        y_bin  = (y_full > thresh).astype(int)
        mask   = np.isfinite(y_full)
        if y_bin[mask].mean() < 0.02 or y_bin[mask].mean() > 0.98:
            continue

        row = dict(morphology=prop, prevalence=float(y_bin[mask].mean()))

        # k-NN (uses full-N topk_s2i, need to remap to probe subset)
        for tag, knn_dict in [('kNN_cross', knn_cross_morph),
                               ('kNN_self',  knn_self_morph)]:
            try:
                scores = knn_dict[prop]
                if n_probe < N:
                    scores = scores[probe_idx]
                row[tag] = roc_auc_score(y_bin[mask], scores[mask])
            except:
                row[tag] = np.nan

        # linear + MLP
        for tag, X in [('lin_cross', sfh_p), ('lin_self', img_p),
                        ('mlp_cross', sfh_p), ('mlp_self', img_p)]:
            model = (LogisticRegression(max_iter=500) if tag.startswith('lin')
                     else MLPRegressor(hidden_layer_sizes=(256, 128),
                                       max_iter=300, early_stopping=True,
                                       random_state=42))
            Xs = StandardScaler().fit_transform(X[mask])
            try:
                y_prob = cross_val_predict(model, Xs, y_bin[mask],
                                           cv=5, method='predict_proba',
                                           n_jobs=-1)[:, 1]
                row[tag] = roc_auc_score(y_bin[mask], y_prob)
            except:
                row[tag] = np.nan

        class_rows.append(row)

    df_class = pd.DataFrame(class_rows).set_index('morphology')
    log.info('\n%s', df_class.to_string(float_format='{:.3f}'.format))

    # 7. write PDF
    log.info('Writing PDF -> %s', args.output)
    with PdfPages(args.output) as pdf:

        # page 1: cosine sim + rank percentile
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        bins = np.linspace(float(min(matched_sim.min(), random_sim.min())) - 0.02,
                           1.0, 80)
        ax = axes[0]
        ax.hist(matched_sim, bins=bins, alpha=0.6, density=True, label='matched pairs')
        ax.hist(random_sim,  bins=bins, alpha=0.6, density=True, label='random pairs')
        ax.axvline(float(matched_sim.mean()), color='tab:blue',   ls='--', lw=1.2)
        ax.axvline(float(random_sim.mean()),  color='tab:orange', ls='--', lw=1.2)
        ax.set_xlabel('Cosine similarity (img . sfh)'); ax.set_ylabel('Density')
        ax.set_title(f'Matched vs random  t={t_stat:.1f}  p={p_val:.1e}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.hist(rank_pctile, bins=50, density=True, alpha=0.8, color='tab:green')
        ax.axvline(0.5, color='grey', ls='--', lw=1, label='random')
        ax.axvline(float(np.median(rank_pctile)), color='tab:red', lw=1.5,
                   label=f'median={np.median(rank_pctile):.3f}')
        ax.set_xlabel('Rank percentile (lower = better)')
        ax.set_ylabel('Density'); ax.set_title('Retrieval rank (img->sfh)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.97,
                f'top  1%: {pct_top1:.1f}%\n'
                f'top  5%: {pct_top5:.1f}%\n'
                f'top 10%: {pct_top10:.1f}%',
                transform=ax.transAxes, va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', fc='w', alpha=0.8))
        fig.suptitle(f'Alignment diagnostics  N={N:,}', fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 2: Recall@k + within-batch rank-1
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        ax.semilogx(ks, rec_i2s, 'o-',  label='img -> sfh', lw=1.5)
        ax.semilogx(ks, rec_s2i, 's--', label='sfh -> img', lw=1.5)
        ax.semilogx(ks, [k / N for k in ks], 'k:', lw=0.8, label='random')
        ax.set_xlabel('k'); ax.set_ylabel('Recall@k')
        ax.set_title('Cross-modal retrieval recall')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
        ax = axes[1]
        vals = [np.mean(r1_i2s), np.mean(r1_s2i), 1 / args.batch_eval]
        bars = ax.bar(['img->sfh', 'sfh->img', 'random'], vals,
                      color=['tab:blue', 'tab:orange', 'grey'], alpha=0.8)
        ax.set_ylabel('Rank-1 accuracy'); ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'Within-batch rank-1  (batch={args.batch_eval},'
                     f' {args.n_batches} trials)')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
        fig.suptitle(f'Retrieval performance  N={N:,}', fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 3: k-NN cross-modal vs in-modality Pearson r
        def grouped_bar(ax, df, col_cross, col_self, xlabel, title):
            labels = list(df.index)
            x      = np.arange(len(labels))
            w      = 0.38
            b1 = ax.barh(x + w/2, df[col_cross], w, label='cross-modal',
                         color='tab:blue', alpha=0.8, edgecolor='white')
            b2 = ax.barh(x - w/2, df[col_self],  w, label='in-modality',
                         color='tab:green', alpha=0.8, edgecolor='white')
            ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(0, color='k', lw=0.8)
            ax.set_xlabel(xlabel); ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7); ax.set_xlim(-0.1, 1.0); ax.grid(axis='x', alpha=0.3)

        n_sfh   = len(df_knn_sfh)
        n_morph = len(df_knn_morph)
        fig_h   = max(5, 0.4 * (n_sfh + n_morph) + 1.5)
        fig, axes = plt.subplots(1, 2, figsize=(13, fig_h))
        grouped_bar(axes[0], df_knn_sfh,   'r_cross', 'r_self',
                    'Pearson r', f'k-NN SFH props (k={args.k})\nimg->sfh vs sfh->sfh')
        grouped_bar(axes[1], df_knn_morph, 'r_cross', 'r_self',
                    'Pearson r', f'k-NN Morph props (k={args.k})\nsfh->img vs img->img')
        fig.suptitle(f'k-NN: cross-modal vs in-modality  N={N:,}', fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 4: probe R2 (linear + MLP, cross vs self) — 4-panel grouped bar
        fig, axes = plt.subplots(2, 2, figsize=(13, fig_h * 1.1))
        grouped_bar(axes[0, 0], df_probe_sfh,   'lin_cross', 'lin_self',
                    'R²', 'Linear probe — SFH props\n(img->sfh vs sfh->sfh)')
        grouped_bar(axes[0, 1], df_probe_morph, 'lin_cross', 'lin_self',
                    'R²', 'Linear probe — Morph props\n(sfh->img vs img->img)')
        grouped_bar(axes[1, 0], df_probe_sfh,   'mlp_cross', 'mlp_self',
                    'R²', 'MLP probe — SFH props\n(img->sfh vs sfh->sfh)')
        grouped_bar(axes[1, 1], df_probe_morph, 'mlp_cross', 'mlp_self',
                    'R²', 'MLP probe — Morph props\n(sfh->img vs img->img)')
        for ax in axes.flatten():
            ax.set_xlabel('R²  (5-fold CV)')
        fig.suptitle(f'Linear & MLP probes: cross-modal vs in-modality  '
                     f'N_probe={n_probe:,}', fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 5: scatter plots — cross-modal k-NN predictions
        SCATTER_PAIRS = [
            (knn_cross_sfh,   'img->sfh [cross]', 'log M*'),
            (knn_cross_sfh,   'img->sfh [cross]', 'log sSFR'),
            (knn_cross_sfh,   'img->sfh [cross]', 'SFH: mean formation epoch'),
            (knn_cross_sfh,   'img->sfh [cross]', 'SFH: log(old/recent)'),
            (knn_cross_morph, 'sfh->img [cross]', 'Sersic n'),
            (knn_cross_morph, 'sfh->img [cross]', 'P(Elliptical)'),
            (knn_cross_morph, 'sfh->img [cross]', 'Axis ratio b/a'),
            (knn_cross_morph, 'sfh->img [cross]', 'P(Late disk)'),
        ]
        SCATTER_PAIRS = [t for t in SCATTER_PAIRS if t[2] in props]
        ncols = 4
        nrows = max(1, (len(SCATTER_PAIRS) + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.2, nrows * 3.0), squeeze=False)
        for ax, (knn_dict, direction, prop) in zip(axes.flatten(), SCATTER_PAIRS):
            y      = props[prop]
            y_pred = knn_dict[prop]
            mask   = np.isfinite(y) & np.isfinite(y_pred)
            pr     = float(stats.pearsonr(y[mask], y_pred[mask])[0])
            ax.scatter(y[mask], y_pred[mask], s=1, alpha=0.3, rasterized=True)
            ax.set_xlabel(f'True {prop}', fontsize=7)
            ax.set_ylabel('k-NN predicted', fontsize=7)
            ax.set_title(f'{prop}  [{direction}]  r={pr:.2f}', fontsize=7)
            ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2)
        for ax in axes.flatten()[len(SCATTER_PAIRS):]:
            ax.set_visible(False)
        fig.suptitle(f'k-NN cross-modal scatter plots  N={N:,}', fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 6: AUC-ROC table
        if not df_class.empty:
            fig, ax = plt.subplots(figsize=(10, max(2, 0.7 * len(df_class) + 2)))
            ax.axis('off')
            tbl = ax.table(
                cellText=df_class.reset_index().round(3).values.tolist(),
                colLabels=['Morphology class'] + list(df_class.columns),
                loc='center', cellLoc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            tbl.auto_set_column_width(list(range(len(df_class.columns) + 1)))
            ax.set_title('Morphology AUC-ROC  (cross-modal vs in-modality)',
                         fontsize=10, pad=15)
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

        # page 7: text summary
        lines = [
            'COSMOS-Web ZooBOT CLIP -- Alignment Evaluation',
            f'N_eval={N:,}   N_probe={n_probe:,}   full={N_FULL:,}   k={args.k}',
            '',
            '-- Cosine similarity (img . sfh) --',
            f'  matched mean={matched_sim.mean():.4f} std={matched_sim.std():.4f}',
            f'  random  mean={random_sim.mean():.4f} std={random_sim.std():.4f}',
            f'  t={t_stat:.1f}  p={p_val:.2e}',
            '',
            '-- Rank percentile (img->sfh) --',
            f'  median={np.median(rank_pctile):.4f}  (0.5=random)',
            f'  top1%={pct_top1:.1f}%  top5%={pct_top5:.1f}%  top10%={pct_top10:.1f}%',
            '',
            f'-- Within-batch rank-1 (batch={args.batch_eval}) --',
            f'  img->sfh={np.mean(r1_i2s):.3f}  sfh->img={np.mean(r1_s2i):.3f}'
            f'  random={1/args.batch_eval:.3f}',
            '',
            f'{"Property":<35s} {"r_cross":>8} {"r_self":>8}'
            f'  {"lin_cross":>9} {"lin_self":>9}'
            f'  {"mlp_cross":>9} {"mlp_self":>9}',
            '-' * 95,
            '  -- SFH props (cross=img->sfh, self=sfh->sfh) --',
        ]
        for lbl in SFH_PROPS:
            rc  = df_knn_sfh.loc[lbl,   'r_cross']   if lbl in df_knn_sfh.index   else np.nan
            rs  = df_knn_sfh.loc[lbl,   'r_self']    if lbl in df_knn_sfh.index   else np.nan
            lc  = df_probe_sfh.loc[lbl, 'lin_cross'] if lbl in df_probe_sfh.index else np.nan
            ls  = df_probe_sfh.loc[lbl, 'lin_self']  if lbl in df_probe_sfh.index else np.nan
            mc  = df_probe_sfh.loc[lbl, 'mlp_cross'] if lbl in df_probe_sfh.index else np.nan
            ms  = df_probe_sfh.loc[lbl, 'mlp_self']  if lbl in df_probe_sfh.index else np.nan
            lines.append(f'  {lbl:<33s} {rc:8.3f} {rs:8.3f}  {lc:9.3f} {ls:9.3f}  {mc:9.3f} {ms:9.3f}')
        lines.append('  -- Morph props (cross=sfh->img, self=img->img) --')
        for lbl in MORPH_PROPS:
            rc  = df_knn_morph.loc[lbl,   'r_cross']   if lbl in df_knn_morph.index   else np.nan
            rs  = df_knn_morph.loc[lbl,   'r_self']    if lbl in df_knn_morph.index   else np.nan
            lc  = df_probe_morph.loc[lbl, 'lin_cross'] if lbl in df_probe_morph.index else np.nan
            ls  = df_probe_morph.loc[lbl, 'lin_self']  if lbl in df_probe_morph.index else np.nan
            mc  = df_probe_morph.loc[lbl, 'mlp_cross'] if lbl in df_probe_morph.index else np.nan
            ms  = df_probe_morph.loc[lbl, 'mlp_self']  if lbl in df_probe_morph.index else np.nan
            lines.append(f'  {lbl:<33s} {rc:8.3f} {rs:8.3f}  {lc:9.3f} {ls:9.3f}  {mc:9.3f} {ms:9.3f}')

        fig, ax = plt.subplots(figsize=(11, max(7, 0.22 * len(lines))))
        ax.axis('off')
        ax.text(0.01, 0.99, '\n'.join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=7.5, fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='#f9f9f9', ec='#cccccc', alpha=0.9))
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)

    log.info('Done. PDF -> %s', args.output)


if __name__ == '__main__':
    main()
