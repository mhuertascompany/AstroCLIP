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
NPZ_PATH = Path('cosmosweb_umap_zoobot_v1.npz')
H5_PATH  = Path('cosmosweb_dataset_v2.h5')

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
# ## 2. Cross-modal rank-1 retrieval accuracy
#
# For each galaxy *i*, find the nearest neighbor of `img_emb[i]` in the SFH
# embedding space.  If alignment is perfect, the answer is always `sfh_emb[i]`
# (rank-1 accuracy = 1.0).  Repeat in the opposite direction.

# %%
print('Computing cross-modal rank-1 retrieval…')

knn1 = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute', n_jobs=-1)

# img → sfh
knn1.fit(sfh_emb)
_, nn_img2sfh = knn1.kneighbors(img_emb)
rank1_img2sfh = (nn_img2sfh[:, 0] == np.arange(N)).mean()

# sfh → img
knn1.fit(img_emb)
_, nn_sfh2img = knn1.kneighbors(sfh_emb)
rank1_sfh2img = (nn_sfh2img[:, 0] == np.arange(N)).mean()

print(f'  Rank-1 accuracy  img→sfh : {rank1_img2sfh:.3f}')
print(f'  Rank-1 accuracy  sfh→img : {rank1_sfh2img:.3f}')
print(f'  (random baseline ≈ {1/N:.2e})')

# Recall@k for several k values
kmax = 50
knn_k = NearestNeighbors(n_neighbors=kmax, metric='cosine', algorithm='brute', n_jobs=-1)
ks    = [1, 5, 10, 20, 50]
recall_img2sfh, recall_sfh2img = [], []

knn_k.fit(sfh_emb)
_, nn = knn_k.kneighbors(img_emb)
for k in ks:
    recall_img2sfh.append((nn[:, :k] == np.arange(N)[:, None]).any(axis=1).mean())

knn_k.fit(img_emb)
_, nn = knn_k.kneighbors(sfh_emb)
for k in ks:
    recall_sfh2img.append((nn[:, :k] == np.arange(N)[:, None]).any(axis=1).mean())

recall_df = pd.DataFrame({
    'k':           ks,
    'Recall@k  img→sfh': recall_img2sfh,
    'Recall@k  sfh→img': recall_sfh2img,
}).set_index('k')
print('\nRetrieval recall:')
print(recall_df.to_string(float_format='{:.3f}'.format))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogx(ks, recall_img2sfh, 'o-', label='img → sfh')
ax.semilogx(ks, recall_sfh2img, 's--', label='sfh → img')
ax.axhline(1/N * np.array(ks).max(), color='grey', lw=0.8, ls=':', label='random@50')
ax.set_xlabel('k'); ax.set_ylabel('Recall@k'); ax.set_title('Cross-modal retrieval recall')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
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
