"""Assess conditional-SFH calibration and residual links to morphology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .umap_zoobot_clip import load_catalog_properties


MORPHOLOGY_TARGETS = {
    'zoobot_smooth_probability': 'P(smooth)',
    'zoobot_featured_probability': 'P(featured/disk)',
    'zoobot_spiral_probability': 'P(spiral arms)',
    'zoobot_bar_probability': 'P(bar)',
    'zoobot_merger_probability': 'P(disturbed/merger)',
}


def cross_validated_r2(features, target, folds=5, seed=42):
    valid = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    if valid.sum() < max(100, folds * 10):
        return np.nan, int(valid.sum())
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    prediction = cross_val_predict(
        model, features[valid], target[valid],
        cv=KFold(folds, shuffle=True, random_state=seed), n_jobs=-1,
    )
    return float(r2_score(target[valid], prediction)), int(valid.sum())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.predictive, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.predictive, 'r') as source:
        ids = np.asarray(source['galaxy_id'], dtype=np.int64)
        rows = np.asarray(source['source_h5_row'], dtype=np.int64)
        condition = np.asarray(source['condition'], dtype=np.float32)
        observed = np.asarray(source['observed_sfh'], dtype=np.float32)
        mean = np.asarray(source['predictive_mean_sfh'], dtype=np.float32)
        p16 = np.asarray(source['predictive_p16_sfh'], dtype=np.float32)
        p84 = np.asarray(source['predictive_p84_sfh'], dtype=np.float32)
        residual = np.asarray(source['residual_sfh'], dtype=np.float32)
        residual_clr = np.asarray(source['residual_clr'], dtype=np.float32)
        time = np.asarray(source['sfh_time_grid'], dtype=np.float32)
        draw_sum_error = float(source.attrs['maximum_absolute_draw_sum_error'])
    with h5py.File(args.dataset, 'r') as source:
        order = np.argsort(rows)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        age_myr = np.asarray(source['sfh_time_norm'][rows[order]])[inverse]
    properties, _ = load_catalog_properties(args.dataset, rows, ids)

    coverage_by_bin = np.mean((observed >= p16) & (observed <= p84), axis=0)
    coverage_by_galaxy = np.mean((observed >= p16) & (observed <= p84), axis=1)
    upper_recent = 100.0 / age_myr
    past_mask = time[None, :] > upper_recent[:, None]
    past_residual_clr = residual_clr * past_mask

    score_rows = []
    baseline = condition
    full_features = np.column_stack([condition, residual_clr])
    past_features = np.column_stack([condition, past_residual_clr])
    for key, label in MORPHOLOGY_TARGETS.items():
        if key not in properties:
            continue
        target = np.asarray(properties[key], dtype=np.float32)
        baseline_r2, count = cross_validated_r2(
            baseline, target, args.folds, args.seed,
        )
        full_r2, _ = cross_validated_r2(
            full_features, target, args.folds, args.seed,
        )
        past_r2, _ = cross_validated_r2(
            past_features, target, args.folds, args.seed,
        )
        score_rows.append({
            'target': key, 'label': label, 'n': count,
            'condition_r2': baseline_r2,
            'condition_plus_full_residual_r2': full_r2,
            'full_residual_delta_r2': full_r2 - baseline_r2,
            'condition_plus_past_residual_r2': past_r2,
            'past_residual_delta_r2': past_r2 - baseline_r2,
        })
    score_path = args.output / 'morphology_residual_scores.csv'
    with score_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)

    pca = PCA(n_components=10, random_state=args.seed)
    residual_pc = pca.fit_transform(residual_clr)
    np.savez_compressed(
        args.output / 'residual_pca.npz', galaxy_id=ids, h5_row=rows,
        residual_pc=residual_pc.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].plot(time, coverage_by_bin, color='tab:blue')
    axes[0, 0].axhline(0.68, color='0.3', ls='--', lw=1, label='nominal 68%')
    axes[0, 0].set(xlabel='Fractional lookback time', ylabel='Empirical coverage',
                   title='Pointwise predictive coverage', ylim=(0, 1))
    axes[0, 0].legend(frameon=False)
    q16, q50, q84 = np.percentile(residual, [16, 50, 84], axis=0)
    axes[0, 1].fill_between(time, q16, q84, color='tab:purple', alpha=0.25)
    axes[0, 1].plot(time, q50, color='tab:purple')
    axes[0, 1].axhline(0, color='0.3', lw=1)
    axes[0, 1].set(xlabel='Fractional lookback time', ylabel='Observed − predicted mean',
                   title='Normalized-SFH residuals')
    labels = [row['label'] for row in score_rows]
    y = np.arange(len(labels))
    axes[1, 0].barh(y - 0.18, [row['full_residual_delta_r2'] for row in score_rows],
                    height=0.35, label='full residual')
    axes[1, 0].barh(y + 0.18, [row['past_residual_delta_r2'] for row in score_rows],
                    height=0.35, label='>100 Myr residual')
    axes[1, 0].axvline(0, color='0.3', lw=1)
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].set(xlabel=r'Cross-validated $\Delta R^2$',
                   title='Morphology information beyond M*, z, SFR100')
    axes[1, 0].legend(frameon=False)
    color_key = 'zoobot_smooth_probability'
    color = properties.get(color_key, np.zeros(len(ids)))
    points = axes[1, 1].scatter(
        residual_pc[:, 0], residual_pc[:, 1], c=color, s=3, alpha=0.45,
        cmap='viridis', rasterized=True,
    )
    axes[1, 1].set(xlabel='Residual PC1', ylabel='Residual PC2',
                   title='Conditional SFH residual space')
    figure.colorbar(points, ax=axes[1, 1], label='P(smooth)')
    figure.tight_layout()
    figure.savefig(args.output / 'conditional_sfh_residual_diagnostics.pdf', dpi=200)
    plt.close(figure)

    report = {
        'predictive_file': str(args.predictive.resolve()),
        'n_galaxies': int(len(ids)),
        'maximum_absolute_draw_sum_error': draw_sum_error,
        'mean_pointwise_68_percent_coverage': float(coverage_by_bin.mean()),
        'median_per_galaxy_68_percent_coverage': float(np.median(coverage_by_galaxy)),
        'mean_residual_l1': float(np.abs(residual).sum(axis=1).mean()),
        'residual_pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'morphology_scores': score_rows,
        'interpretation': (
            'delta R2 measures linear morphology information in conditional SFH '
            'residuals beyond mass, redshift, and SFH-derived SFR100.'
        ),
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
