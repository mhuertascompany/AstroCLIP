# %% [markdown]
# # COSMOS-Web CLIP — Alignment Evaluation
#
# Evaluates how well image and SFH embeddings are aligned by predicting
# physical properties cross-modally via k-NN retrieval and linear probes.
#
# **Required files (download from candide):**
# - `cosmosweb_umap_zoobot_v1.npz`  — embeddings + catalogue properties
# - `cosmosweb_dataset_v2.h5`       — HDF5 for SFH shape descriptors
#
# All computation runs locally (CPU, ~1–2 min for 50 k galaxies).

# %% [markdown]
# ## 0. Setup

# %%
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

# ── paths — edit these ────────────────────────────────────────────────────────
NPZ_PATH = Path('/Users/marchuertascompany/Documents/python_scripts/cosmosweb_SFHs/cosmosweb_umap_zoobot_v1.npz')
H5_PATH  = Path('/Users/marchuertascompany/Documents/python_scripts/cosmosweb_SFHs/cosmosweb_dataset_v2.h5')

# ── SFH grid (must match prepare_dataset.py) ──────────────────────────────────
SFH_N_BINS = 50
SFH_EPS    = 1e-10
T_FRAC     = np.linspace(0, 1, SFH_N_BINS)   # bin 0 = now, bin 49 = Big Bang

# %% [markdown]
# ## 1. Load data

# %%
print('Loading npz…')
npz = np.load(NPZ_PATH, allow_pickle=True)

img_emb = npz['img_emb'].astype(np.float64)   # (N, D)
sfh_emb = npz['sfh_emb'].astype(np.float64)   # (N, D)
N, D    = img_emb.shape
print(f'  {N} galaxies,  embedding dim = {D}')

# ── catalogue properties from npz ────────────────────────────────────────────
props = {}
for key, label in [
    ('zfinal',             'Redshift z'),
    ('sersic',             'Sersic n'),
    ('radius_sersic',      'Sersic radius [px]'),
    ('axratio_sersic',     'Axis ratio b/a'),
    ('log_mass',           'log M★'),
    ('log_sfr',            'log SFR'),
    ('log_ssfr',           'log sSFR'),
    ('age_form',           'Formation age [Myr]'),
    ('log_sfr_inst',       'log SFR_inst'),
    ('log_sfr_100myr',     'log SFR_100Myr'),
    ('sfr_mass_vector_dir',  'SFR–M★ dir.'),
    ('sfr_mass_vector_norm', 'SFR–M★ norm'),
    ('family_elliptical',  'P(Elliptical)'),
    ('family_s0',          'P(S0)'),
    ('family_early_disk',  'P(Early disk)'),
    ('family_late_disk',   'P(Late disk)'),
]:
    if key in npz:
        props[label] = npz[key].astype(np.float64)

print(f'  Catalogue properties available: {list(props.keys())}')

# ── SFH shape descriptors from HDF5 ──────────────────────────────────────────
h5_indices = npz['h5_indices'].astype(int)
print('Loading SFH vectors from HDF5…')
with h5py.File(H5_PATH, 'r') as f:
    sfh_log = f['sfh'][list(h5_indices)].astype(np.float64)   # (N, 50)

sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
sfr_sum = sfr.sum(axis=1, keepdims=True)
sfr = sfr / np.where(sfr_sum > 0, sfr_sum, 1.0)

t = T_FRAC[np.newaxis, :]
props['SFH: log(old/recent)']      = sfh_log[:, -1] - sfh_log[:, 0]
props['SFH: mean formation epoch'] = (sfr * t).sum(axis=1)
props['SFH: f(recent 20%)']        = sfr[:, :SFH_N_BINS // 5].sum(axis=1)
props['SFH: peak lookback t_frac'] = T_FRAC[np.argmax(sfr, axis=1)]
print(f'  SFH shape descriptors added. Total properties: {len(props)}')

# ── define property groups ────────────────────────────────────────────────────
MORPH_PROPS = [
    'Sersic n', 'Sersic radius [px]', 'Axis ratio b/a',
    'P(Elliptical)', 'P(S0)', 'P(Early disk)', 'P(Late disk)',
]
SFH_PROPS = [
    'log M★', 'log SFR', 'log sSFR',
    'Formation age [Myr]', 'log SFR_inst', 'log SFR_100Myr',
    'SFR–M★ dir.', 'SFR–M★ norm',
    'SFH: log(old/recent)', 'SFH: mean formation epoch',
    'SFH: f(recent 20%)', 'SFH: peak lookback t_frac',
]
MORPH_PROPS = [p for p in MORPH_PROPS if p in props]
SFH_PROPS   = [p for p in SFH_PROPS   if p in props]

# %% [markdown]
# ## 2. Alignment diagnostics
#
# Rank-1 accuracy over the *full* dataset is an extremely strict metric:
# the model was trained to distinguish **batch_size=128 negatives**, not N≈100k.
# The right questions are:
#
# 1. **Matched vs random cosine similarity** — are (img[i], sfh[i]) pairs more
#    similar than random cross-modal pairs?
# 2. **Rank percentile** — where does the correct SFH rank in the sorted similarity
#    list for each image query? (0 = best, 1 = worst)
# 3. **Within-batch retrieval** — rank-1 evaluated in random batches of 128,
#    matching training conditions.
# 4. **Recall@k** at larger k to see if there is any signal.

# %%
# ── 2a. Matched vs random cosine similarity ───────────────────────────────────
matched_sim  = (img_emb * sfh_emb).sum(axis=1)           # cos-sim of each pair

rng = np.random.default_rng(42)
rand_idx     = rng.permutation(N)
random_sim   = (img_emb * sfh_emb[rand_idx]).sum(axis=1) # shuffled → random pairs

print('Cosine similarity  (img, sfh):')
print(f'  matched pairs  : mean={matched_sim.mean():.4f}  std={matched_sim.std():.4f}')
print(f'  random  pairs  : mean={random_sim.mean():.4f}  std={random_sim.std():.4f}')
t_stat, p_val = stats.ttest_ind(matched_sim, random_sim)
print(f'  t-test  t={t_stat:.1f}  p={p_val:.2e}')

fig, ax = plt.subplots(figsize=(5, 3.5))
bins = np.linspace(-0.2, 1.0, 80)
ax.hist(matched_sim, bins=bins, alpha=0.6, label='matched pairs',  density=True)
ax.hist(random_sim,  bins=bins, alpha=0.6, label='random pairs',   density=True)
ax.axvline(matched_sim.mean(), color='tab:blue',   ls='--', lw=1.2)
ax.axvline(random_sim.mean(),  color='tab:orange', ls='--', lw=1.2)
ax.set_xlabel('Cosine similarity'); ax.set_ylabel('Density')
ax.set_title('Matched vs random cross-modal similarity')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cosine_sim_dist.pdf', dpi=150); plt.show()

# %%
# ── 2b. Rank percentile of the correct SFH embedding ─────────────────────────
# For a random subsample (full N²  matrix is too large):
N_SAMPLE = min(5_000, N)
sample   = rng.choice(N, N_SAMPLE, replace=False)

img_s = img_emb[sample]   # (N_SAMPLE, D)
sfh_s = sfh_emb[sample]   # (N_SAMPLE, D)

sim_matrix   = img_s @ sfh_emb.T          # (N_SAMPLE, N) — query against ALL sfh
correct_sims = sim_matrix[np.arange(N_SAMPLE), sample]   # diagonal scores

# rank percentile: fraction of gallery items ABOVE the correct match
rank_pctile = (sim_matrix > correct_sims[:, None]).mean(axis=1)   # 0=best, 1=worst
median_pctile = np.median(rank_pctile)
print(f'\nRank percentile of correct pair (img→sfh, subsample={N_SAMPLE}):')
print(f'  median = {median_pctile:.4f}  (0=always top, 0.5=random)')
print(f'  mean   = {rank_pctile.mean():.4f}')
print(f'  % in top 1%  : {(rank_pctile < 0.01).mean()*100:.1f}%')
print(f'  % in top 5%  : {(rank_pctile < 0.05).mean()*100:.1f}%')
print(f'  % in top 10% : {(rank_pctile < 0.10).mean()*100:.1f}%')

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(rank_pctile, bins=50, density=True, alpha=0.8)
ax.axvline(0.5, color='grey', ls='--', lw=1, label='random baseline')
ax.axvline(median_pctile, color='tab:red', ls='-', lw=1.5,
           label=f'median={median_pctile:.3f}')
ax.set_xlabel('Rank percentile of correct pair\n(lower = better)')
ax.set_ylabel('Density'); ax.set_title('Retrieval rank distribution')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rank_percentile.pdf', dpi=150); plt.show()

# %%
# ── 2c. Within-batch retrieval  (matches training conditions) ─────────────────
BATCH_EVAL = 128
N_BATCHES  = 200
rank1_batch_img2sfh, rank1_batch_sfh2img = [], []

for _ in range(N_BATCHES):
    idx  = rng.choice(N, BATCH_EVAL, replace=False)
    iemb = img_emb[idx]   # (128, D)
    semb = sfh_emb[idx]   # (128, D)

    # img → sfh: row i should be most similar to column i
    logits_i2s = iemb @ semb.T   # (128, 128)
    preds_i2s  = logits_i2s.argmax(axis=1)
    rank1_batch_img2sfh.append((preds_i2s == np.arange(BATCH_EVAL)).mean())

    logits_s2i = semb @ iemb.T
    preds_s2i  = logits_s2i.argmax(axis=1)
    rank1_batch_sfh2img.append((preds_s2i == np.arange(BATCH_EVAL)).mean())

r1_i2s = np.mean(rank1_batch_img2sfh)
r1_s2i = np.mean(rank1_batch_sfh2img)
print(f'\nWithin-batch rank-1  (batch={BATCH_EVAL}, {N_BATCHES} random batches):')
print(f'  img→sfh : {r1_i2s:.3f}  (random baseline = {1/BATCH_EVAL:.3f})')
print(f'  sfh→img : {r1_s2i:.3f}')

# %%
# ── 2d. Recall@k across a wider range ────────────────────────────────────────
# Use the subsample similarity matrix already computed above
ks = [1, 5, 10, 50, 100, 500, 1000]
ks = [k for k in ks if k <= N]
recall_img2sfh, recall_sfh2img = [], []

for k in ks:
    top_k = np.argsort(sim_matrix, axis=1)[:, -k:]
    recall_img2sfh.append((top_k == sample[:, None]).any(axis=1).mean())

# sfh → img direction
sim_s2i = sfh_s @ img_emb.T   # (N_SAMPLE, N)
for k in ks:
    top_k = np.argsort(sim_s2i, axis=1)[:, -k:]
    recall_sfh2img.append((top_k == sample[:, None]).any(axis=1).mean())

recall_df = pd.DataFrame({
    'k': ks,
    'Recall@k img→sfh': recall_img2sfh,
    'Recall@k sfh→img': recall_sfh2img,
    'Random baseline':  [k / N for k in ks],
}).set_index('k')
print('\nRetrieval recall (subsampled queries):')
print(recall_df.to_string(float_format='{:.4f}'.format))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogx(ks, recall_img2sfh,           'o-',  label='img → sfh')
ax.semilogx(ks, recall_sfh2img,           's--', label='sfh → img')
ax.semilogx(ks, [k / N for k in ks],     'k:',  lw=0.8, label='random')
ax.set_xlabel('k'); ax.set_ylabel('Recall@k')
ax.set_title('Cross-modal retrieval recall')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('recall_at_k.pdf', dpi=150); plt.show()

# %% [markdown]
# ## 3. k-NN property prediction  (cross-modal)
#
# Given `img_emb[i]`, find its *k* nearest neighbours in SFH embedding space
# and predict the SFH/SED property as their mean.  Vice-versa for morphology.

# %%
K_PRED = 10   # number of neighbours for prediction

def knn_predict(query_emb, key_emb, prop_values, k=K_PRED):
    """
    For each row in query_emb find k nearest rows in key_emb;
    return predicted values (mean of neighbours) and valid mask.
    """
    knn = NearestNeighbors(n_neighbors=k, metric='cosine',
                           algorithm='brute', n_jobs=-1)
    knn.fit(key_emb)
    _, idx = knn.kneighbors(query_emb)     # (N, k)
    predicted = prop_values[idx].mean(axis=1)
    return predicted

def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20:
        return dict(pearson_r=np.nan, spearman_r=np.nan, r2=np.nan, mae=np.nan, n=mask.sum())
    yt, yp = y_true[mask], y_pred[mask]
    pr, _  = stats.pearsonr(yt, yp)
    sr, _  = stats.spearmanr(yt, yp)
    r2     = r2_score(yt, yp)
    mae    = np.abs(yt - yp).mean()
    return dict(pearson_r=pr, spearman_r=sr, r2=r2, mae=mae, n=mask.sum())

print(f'k-NN property prediction  (k={K_PRED})…')

rows_sfh_from_img, rows_morph_from_sfh = [], []

# ── SFH/SED properties predicted from image embeddings ───────────────────────
for label in SFH_PROPS:
    y = props[label]
    y_pred = knn_predict(img_emb, sfh_emb, y, k=K_PRED)
    m = regression_metrics(y, y_pred)
    rows_sfh_from_img.append({'property': label, **m})

# ── Morphology predicted from SFH embeddings ─────────────────────────────────
for label in MORPH_PROPS:
    y = props[label]
    y_pred = knn_predict(sfh_emb, img_emb, y, k=K_PRED)
    m = regression_metrics(y, y_pred)
    rows_morph_from_sfh.append({'property': label, **m})

df_sfh_from_img  = pd.DataFrame(rows_sfh_from_img).set_index('property')
df_morph_from_sfh = pd.DataFrame(rows_morph_from_sfh).set_index('property')

print('\n── SFH / SED properties predicted from image embeddings ──')
print(df_sfh_from_img[['pearson_r', 'spearman_r', 'r2', 'mae']].to_string(float_format='{:.3f}'.format))
print('\n── Morphology predicted from SFH embeddings ──')
print(df_morph_from_sfh[['pearson_r', 'spearman_r', 'r2', 'mae']].to_string(float_format='{:.3f}'.format))

# %% [markdown]
# ### 3a. Scatter plots for selected properties

# %%
SCATTER_PAIRS = [
    # (query_emb_label, key_emb_label, property_label)
    ('Image emb.',  'SFH emb.',  'log M★'),
    ('Image emb.',  'SFH emb.',  'log sSFR'),
    ('Image emb.',  'SFH emb.',  'SFH: mean formation epoch'),
    ('Image emb.',  'SFH emb.',  'SFH: log(old/recent)'),
    ('SFH emb.',    'Image emb.','Sersic n'),
    ('SFH emb.',    'Image emb.','P(Elliptical)'),
    ('SFH emb.',    'Image emb.','Axis ratio b/a'),
    ('SFH emb.',    'Image emb.','P(Late disk)'),
]

emb_map = {'Image emb.': img_emb, 'SFH emb.': sfh_emb}

ncols = 4
nrows = (len(SCATTER_PAIRS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0))

for ax, (q_label, k_label, prop) in zip(axes.flatten(), SCATTER_PAIRS):
    if prop not in props:
        ax.set_visible(False); continue
    y = props[prop]
    y_pred = knn_predict(emb_map[q_label], emb_map[k_label], y, k=K_PRED)
    mask = np.isfinite(y) & np.isfinite(y_pred)
    pr, _ = stats.pearsonr(y[mask], y_pred[mask])
    ax.scatter(y[mask], y_pred[mask], s=1, alpha=0.3, rasterized=True)
    ax.set_xlabel(f'True {prop}', fontsize=7)
    ax.set_ylabel(f'Predicted (k-NN)', fontsize=7)
    ax.set_title(f'{prop}\n{q_label}→{k_label}  r={pr:.2f}', fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes.flatten()[len(SCATTER_PAIRS):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('knn_scatter.pdf', dpi=150); plt.show()

# %% [markdown]
# ## 4. Linear probe  (cross-modal)
#
# Train a Ridge regressor on one modality's embeddings to predict properties
# from the other modality.  5-fold CV R² gives a clean upper bound on how
# much cross-modal information is linearly decodable.

# %%
def linear_probe(X, y, cv=5, alpha=1.0):
    """5-fold CV R² of Ridge regression X → y."""
    mask = np.isfinite(y)
    if mask.sum() < 50:
        return np.nan
    Xm, ym = X[mask], y[mask]
    scaler = StandardScaler()
    Xm = scaler.fit_transform(Xm)
    scores = cross_val_score(Ridge(alpha=alpha), Xm, ym,
                             cv=cv, scoring='r2', n_jobs=-1)
    return float(scores.mean())

print('Linear probe (5-fold CV R²)…  [this may take ~1 min]')

lp_sfh_from_img, lp_morph_from_sfh = {}, {}

for label in SFH_PROPS:
    lp_sfh_from_img[label] = linear_probe(img_emb, props[label])
    print(f'  img→{label:<35s}  R²={lp_sfh_from_img[label]:.3f}')

for label in MORPH_PROPS:
    lp_morph_from_sfh[label] = linear_probe(sfh_emb, props[label])
    print(f'  sfh→{label:<35s}  R²={lp_morph_from_sfh[label]:.3f}')

# %% [markdown]
# ## 5. Morphology classification  (AUC-ROC)
#
# For each morphological class, define a binary label (P > 0.5) and measure
# how well the *SFH* embedding predicts it with k-NN and with a linear probe.

# %%
MORPH_CLASSES = {
    'Elliptical':  ('P(Elliptical)',  0.5),
    'S0':          ('P(S0)',          0.5),
    'Early disk':  ('P(Early disk)',  0.5),
    'Late disk':   ('P(Late disk)',   0.5),
}

class_rows = []
for name, (prop, thresh) in MORPH_CLASSES.items():
    if prop not in props:
        continue
    y_score = props[prop]
    y_bin   = (y_score > thresh).astype(int)
    mask    = np.isfinite(y_score)
    frac    = y_bin[mask].mean()
    if frac < 0.02 or frac > 0.98:
        continue   # degenerate class

    # k-NN score: proportion of Elliptical among k SFH neighbours
    knn_score = knn_predict(sfh_emb, img_emb, y_score.astype(float), k=K_PRED)
    try:
        auc_knn = roc_auc_score(y_bin[mask], knn_score[mask])
    except Exception:
        auc_knn = np.nan

    # Linear probe AUC (train on sfh_emb, predict binary class)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Xm = StandardScaler().fit_transform(sfh_emb[mask])
    try:
        y_prob = cross_val_predict(
            LogisticRegression(max_iter=500, C=1.0), Xm, y_bin[mask],
            cv=5, method='predict_proba', n_jobs=-1
        )[:, 1]
        auc_lr = roc_auc_score(y_bin[mask], y_prob)
    except Exception:
        auc_lr = np.nan

    class_rows.append(dict(
        morphology=name, prevalence=frac,
        AUC_kNN=auc_knn, AUC_linear=auc_lr,
    ))

df_class = pd.DataFrame(class_rows).set_index('morphology')
print('\n── Morphology classification from SFH embeddings (AUC-ROC) ──')
print(df_class.to_string(float_format='{:.3f}'.format))

# %% [markdown]
# ## 6. Summary figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ── panel 1: k-NN Pearson r heatmap ──────────────────────────────────────────
all_labels = list(df_sfh_from_img.index) + list(df_morph_from_sfh.index)
all_r      = (list(df_sfh_from_img['pearson_r']) +
              list(df_morph_from_sfh['pearson_r']))
colours    = (['tab:blue'] * len(df_sfh_from_img) +
              ['tab:orange'] * len(df_morph_from_sfh))

ax = axes[0]
y_pos = np.arange(len(all_labels))
bars = ax.barh(y_pos, all_r, color=colours, alpha=0.8, edgecolor='white')
ax.set_yticks(y_pos); ax.set_yticklabels(all_labels, fontsize=7)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Pearson r  (k-NN cross-modal prediction)')
ax.set_title(f'k-NN  (k={K_PRED})', fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='tab:blue',   label='img→SFH props'),
                   Patch(color='tab:orange', label='sfh→Morph props')],
          fontsize=7, loc='lower right')
ax.set_xlim(-0.1, 1.0); ax.grid(axis='x', alpha=0.3)

# ── panel 2: linear probe R² ─────────────────────────────────────────────────
lp_labels = list(lp_sfh_from_img.keys()) + list(lp_morph_from_sfh.keys())
lp_r2     = list(lp_sfh_from_img.values()) + list(lp_morph_from_sfh.values())
lp_cols   = (['tab:blue'] * len(lp_sfh_from_img) +
             ['tab:orange'] * len(lp_morph_from_sfh))

ax = axes[1]
y_pos = np.arange(len(lp_labels))
ax.barh(y_pos, lp_r2, color=lp_cols, alpha=0.8, edgecolor='white')
ax.set_yticks(y_pos); ax.set_yticklabels(lp_labels, fontsize=7)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('R²  (5-fold CV Ridge regression)')
ax.set_title('Linear probe', fontsize=9)
ax.set_xlim(-0.1, 1.0); ax.grid(axis='x', alpha=0.3)

# ── panel 3: retrieval recall@k ──────────────────────────────────────────────
ax = axes[2]
ax.semilogx(ks, recall_img2sfh, 'o-', label='img → sfh', lw=1.5)
ax.semilogx(ks, recall_sfh2img, 's--', label='sfh → img', lw=1.5)
ax.set_xlabel('k'); ax.set_ylabel('Recall@k')
ax.set_title('Cross-modal retrieval recall', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.suptitle('COSMOS-Web ZooBOT CLIP — Alignment Evaluation', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('alignment_summary.pdf', dpi=150, bbox_inches='tight')
plt.show()
print('\nSaved: recall_at_k.pdf, knn_scatter.pdf, alignment_summary.pdf')
