# Euclid SFH pilot sample

Run on candide from the repository root, in an environment with NumPy, h5py,
Astropy, and the dependencies of the existing `get_paths`/`load_table` helpers:

```bash
python -m euclid.sample_edfn_sfhs \
    --output edfn_10k \
    --n 10000 \
    --seed 42
```

Choose a new or empty output directory. Defaults use:

- Helpers: `/home/wozny/jobs/These/DR1_science/SFH/utils`
- Photometric catalog: `get_file_cat(field="EDFN", cat_name="clean_photo_phz", phot_type="2fwhm_aper")`, loaded with `open_table`.
- SFHs: HDF5 files under `/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs`.
- ID column: `object_id` (override with `--id-column` for the photometric catalog).

To submit this preparation as a SLURM job on candide:

```bash
sbatch euclid/slurm_sample_edfn_sfhs.sh
```

For the 100,000-object training sample, submit the separate large job:

```bash
sbatch euclid/slurm_sample_edfn_sfhs_100k.sh
```

It writes to `/n03data/huertas/euclid/sfh_clip/edfn_100k/`, uses the same seed
and `EDFN_2fwhm_aper.h5` input as the pilot, and has a 24-hour time limit. It
fails cleanly if fewer than 100,000 photometric-catalog objects have SFHs.

The job uses node `n03`, 4 CPUs, 64 GB RAM, and a 6-hour limit, with the same
`cosmos_visual` conda environment as the COSMOS-Web preparation scripts.
It writes the 10,000-object sample (seed 42) to
`/n03data/huertas/euclid/sfh_clip/edfn_10k/`. Job logs are written to
`euclid/sample_edfn_sfhs_<jobid>.out` and `.err` in the candide checkout.
The environment must also provide the dependencies of Wozny's catalog helpers.
The job explicitly reads
`ready_to_use_sfhs/EDFN_2fwhm_aper.h5`, matching the `2fwhm_aper` photometric
catalog. It does not scan or combine the alternate `EDFN.h5` product.
Sampler options can be passed after the script name, for example:

```bash
sbatch euclid/slurm_sample_edfn_sfhs.sh --output /path/to/new_sample
```

The output directory may already exist if it is empty; otherwise the sampler
creates it. Existing files are never overwritten. If an earlier run left a
partial sample, choose another `--output` directory.

The script identifies EDFN membership through the photometric catalog, matches
its IDs to the SFH files, and uniformly samples 10,000 matched galaxies without
replacement. This samples the **SFH-available subset** of the clean EDFN catalog;
it adds no redshift, mass, morphology, or SFH quality cuts. It reports the number
of catalog objects with and without SFHs, and fails if fewer than 10,000 match.
The seed and sorted source filenames make selection reproducible for unchanged
inputs. Output rows are sorted by source row for efficient HDF5 access.

If the directory contains alternative SFH fits for the same IDs, the script
stops rather than choosing between them. Supply one consistent set explicitly:

```bash
python -m euclid.sample_edfn_sfhs \
    --sfh-files /path/to/catalog_part1.h5 /path/to/catalog_part2.h5 \
    --output /path/to/edfn_10k
```

Outputs:

- `catalog.fits`: all selected photometric columns, including coordinates if
  present, plus `field_catalog_row`, `sfh_file`, `sfh_row`, `sfh_source_file`, and
  `sfh_source_row`. `sfh_row` is the zero-based row in the exported subset;
  `sfh_source_row` refers to the original file. File paths in `sfh_file` are
  relative to the output directory.
- `object_ids.csv`: exact object IDs and SFH file/row references for matching.
- `sfh_000.h5`, etc.: one subset per contributing input file, readable with
  `SFHCatalog`, preserving all SFH realizations, the shared native `age` grid,
  derived quantities, and dataset attributes. No SFH rescaling is applied.
- `sfh_000.csv`, etc.: scalar SFH metadata and original row indices.

SFH files must have the flat schema documented in `sfh_catalog.py`: `sfh` has
shape `(galaxy, realization, time)`, `age` is a shared time grid, and other
non-scalar datasets have galaxy as their first dimension. The exporter reads
SFHs in batches (`--batch-size 128` by default).

For an input already restricted to the desired field, the simpler exporter is:

```bash
python -m euclid.sample_sfh_catalog \
    --catalog /path/to/EDFN.h5 --output /path/to/edfn_10k.h5 --n 10000 --seed 42
```

The workflows have been checked against synthetic catalogs locally; the
candide helper return type, source schema, and actual field overlap still need
verification on the first real run.

## Preprocess SFHs for CLIP

The Euclid catalogs store 50 posterior SFH realizations per galaxy on a common
physical lookback-time grid. Convert them to the representation used by the
COSMOS-Web SFH encoder with:

```bash
python -m euclid.preprocess_sfhs \
    --input /path/to/edfn_100k/sfh_000.h5 \
    --output /path/to/edfn_100k/sfh_clip_100k.h5
```

For each galaxy, the converter uses cumulative mass fractions to rebin every
one of the 50 posterior realizations onto a uniform grid in fractional
lookback time (`lookback time / age of the Universe at the galaxy redshift`).
It normalizes each rebinned realization to sum to one and stores
`log10(weight + 1e-10)`. This removes absolute SFH amplitude while preserving
shape and posterior uncertainty. Cumulative rebinning conserves the integrated
weights and prevents narrow bursts from being missed between grid points. By
default, the output retains the native number of time bins, giving `sfh` shape
`(galaxy, 250)` for the Euclid DR1 catalogs. `sfh_time_grid` contains the shared
`[0, 1]` grid, and `sfh_time_norm` contains the age of the Universe in Myr for
each object. Set the SFH encoder input dimension to 250 for Euclid training.
`--n-bins` remains available for experiments at another resolution.

The `sfh` dataset is the median of the processed realizations and remains the
deterministic encoder input. `sfh_realizations` has shape
`(galaxy, 50, 250)` and can be sampled along its second axis during training so
the SFH encoder sees the posterior uncertainty. `sfh_p16` and `sfh_p84` provide
precomputed diagnostic bounds. All four datasets use the same log10 convention.
`sfh_realization_valid` identifies realizations that contain mass within the
age of the Universe implied by the catalog redshift and have nonzero native
weight; training samples only valid entries. Invalid entries in
`sfh_realizations` are NaN rather than a fabricated curve. A zero-weight draw
does not invalidate a galaxy when another posterior draw is usable. The
converter fails only when a galaxy has no valid realization.
`sfh_retained_mass_fraction` records how much native mass was inside the
physical range before each realization was renormalized.

The output retains `object_id` and all other scalar row metadata, and provides
the aliases `galaxy_id` and `redshift`. These IDs can be joined to the cutout
manifest without relying on row order. The original 50 realizations and
two-dimensional derived posterior quantities remain in the source HDF5.

For the 100,000-object candide sample:

```bash
sbatch euclid/slurm_preprocess_edfn_sfhs_100k.sh
```

This reads `edfn_100k/sfh_000.h5` and writes
`edfn_100k/sfh_clip_100k.h5`. The converter refuses to overwrite an existing
output and writes through a temporary file so an interrupted job cannot leave a
partial product at the final path. To use different paths, pass the input and
output after the script name:

```bash
sbatch euclid/slurm_preprocess_edfn_sfhs_100k.sh \
    /path/to/input.h5 /path/to/new_output.h5
```

## Bulk mosaic cutouts on ESA Datalabs

`bulk_sfh_cutouts.py` follows **Cutouts_v3.ipynb**, using Astroquery to find
`dr1.mosaic_product` records and Astropy `Cutout2D` to extract images directly
from the mounted Datalabs data volume. Run in the **EUCLID-TOOLS** environment
with `numpy`, `astropy`, and `astroquery` supporting `EuclidClass(environment='IDR')`.
It does not require `n_utils.py`, Cutana, or euclidkit.

Copy the sample's **`catalog.fits`** from candide into your Datalabs workspace.
Only the catalog is needed there; the SFH HDF5 files can stay on candide.
The catalog must include object IDs and coordinates. The default coordinate
aliases are `right_ascension`/`declination`, `ra`/`dec`, or `ra_deg`/`dec_deg`,
matched case-insensitively. Use `--ra-column` and `--dec-column` to override.

From the repository root on Datalabs:

```bash
python -m euclid.bulk_sfh_cutouts \
    --sample /media/user/edfn_10k/catalog.fits \
    --output /media/user/edfn_10k_cutouts \
    --bands VIS \
    --size-arcsec 10
```

The script also works standalone if you copy only `bulk_sfh_cutouts.py`:

```python
# Run this in an EUCLID-TOOLS notebook cell; adjust the script location.
%run /media/user/bulk_sfh_cutouts.py --sample /media/user/edfn_10k/catalog.fits --output /media/user/edfn_10k_cutouts --bands VIS --size-arcsec 10
```

Astroquery prompts for archive login. An existing Astroquery credentials file
can instead be supplied with `--credentials-file /path/to/credentials`.
The upload contains only sample row numbers, exact integer IDs, and coordinates;
it is a temporary TAP query upload, not a persistent archive user table.

Defaults select **DEEP processing-mode mosaics**, and a **10-arcsecond square
side**, matching the last cutout example in the notebook. The angular size is
not a radius. Pixels are kept at each band's **native resolution and units**,
without resizing, normalization, stretching, or PSF matching. Thus VIS and NISP
outputs need not have identical pixel dimensions. To request all four bands:

```bash
python -m euclid.bulk_sfh_cutouts \
    --sample /media/user/edfn_10k/catalog.fits \
    --output /media/user/edfn_10k_cutouts_all_bands \
    --bands VIS NIR-Y NIR-J NIR-H \
    --size-arcsec 10
```

The script uploads batches of 1,000 sources, finds overlapping mosaic footprints,
and processes cutouts grouped by mosaic file. For multiple matches it tries the
mosaic whose center is closest first, with filename as a deterministic tie-break.
If that candidate is unreadable, crosses an image edge, or contains nonfinite
pixels, it tries the next candidate. Cutouts are not padded or silently trimmed.
All candidates are retained in `mosaic_matches.ecsv`; use `--release-name` to
restrict to a particular release, or `--processing-mode WIDE` if needed.

Outputs:

- `cutouts/VIS/<object_id>.fits` (and corresponding band directories): native
  science pixels with updated WCS, units, object ID, band, and mosaic provenance.
- `manifest.csv`: one row per requested galaxy/band, including the cutout path,
  SFH file/row references, original sample row, selected mosaic, dimensions, and
  status (`written`, `existing`, `no_mosaic`, or `failed`). SFH file references
  remain relative to the original SFH sample directory, not the cutout directory.
- `mosaic_matches.ecsv`: every candidate mosaic returned by the archive.
- `queries/`: ADQL and cached results for each source batch.
- `run.json`: sample checksum and processing settings, used to validate restarts.

If any cutouts are missing, the command exits unsuccessfully after saving the
manifest and all successful images. A coverage failure does not remove the
galaxy from the manifest. Retry the **same command with `--resume`** to reuse
completed queries and verified FITS outputs. Changing the sample, bands, size,
or query settings requires a new output directory.

Each uncached archive batch is attempted five times by default, with an
exponential delay capped at 30 seconds. Adjust this with `--query-retries` and
`--retry-delay`. If an archive login expires during a long lookup, restart the
same command with `--resume`; it logs in again and queries only uncached batches.

For an initial check, add `--limit 100` and use a separate output directory.
`--query-only` saves mosaic matches without extracting images; use the same
command with `--resume` and without `--query-only` to extract them later.

To transfer the results back to candide or your laptop, package the output and
download the archive from the Datalabs file browser:

```bash
tar -cf /media/user/edfn_10k_cutouts.tar -C /media/user edfn_10k_cutouts
```

Offline validation uses synthetic FITS mosaics to check WCS and pixel fidelity,
large object IDs, SFH links, overlapping-tile fallback, missing images,
multiple bands, query batching, and restart behavior:

```bash
python -m unittest euclid.test_bulk_sfh_cutouts
```

The actual authenticated IDR queries and mounted mosaic files still need a
first run on Datalabs. The implementation uses the table/column names in the
supplied notebook and the documented
[Astroquery Euclid API](https://astroquery.readthedocs.io/en/latest/api/astroquery.esa.euclid.EuclidClass.html).

## Prepare VIS cutouts for ZooBot

The SFH selection catalog is based on `clean_photo_phz`, which does not include
the `SEGMENTATION_AREA` and `ELLIPTICITY` columns required by the reference
Euclid `R_MAX` regression. Fetch those exact values by `OBJECT_ID` from the MER
deep-survey catalog on Datalabs before transferring the metadata file to
candide:

```bash
cd /media/user/python/AstroCLIP
/opt/miniforge/envs/euclid-tools/bin/python -m euclid.fetch_mer_morphology \
    --sample /home/mhuertas/my_workspace/sfh_edfn100k/catalog_sfh_100k.fits \
    --output /home/mhuertas/my_workspace/sfh_edfn100k/morphology_catalog_sfh_100k.fits \
    --batch-size 1000 \
    --resume
```

The query uses `catalogue.mer_catalogue_deep_survey` and requires a complete,
one-to-one match for all object IDs. Completed batches are cached next to the
output, so `--resume` continues after a TAP failure. Transfer the resulting
`morphology_catalog_sfh_100k.fits` to:

```text
/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_edfn100k/morphology_catalog_sfh_100k.fits
```

After transferring the complete Datalabs cutout directory to candide, convert
the successful VIS FITS cutouts into the JPEG layout used by the existing
ZooBot CLIP loader:

```bash
python -m euclid.prepare_zoobot_cutouts \
    --cutout-root /path/to/transferred_cutout_directory \
    --catalog /path/to/morphology_catalog_sfh_100k.fits \
    --output /path/to/zoobot_stamps \
    --band VIS \
    --image-size 224 \
    --workers 8
```

`--cutout-root` must contain the original `manifest.csv` and `cutouts/VIS/`
tree. `--catalog` is the matching MER morphology table exported on Datalabs.
Only rows
with source status `written` or `existing` are converted. Following
`morphology_utils.py`, the converter estimates `R_MAX` in VIS pixels from
`SEGMENTATION_AREA`, `KRON_RADIUS`, and `ELLIPTICITY`. These must be exact MER
catalog values; the converter does not substitute a size proxy. It then crops
each source to a square of half-width `R_MAX`, applies `arcsinh(flux * 100)`,
clips at the 99.85th percentile, and bicubically resizes it to 224 pixels. The
result is saved as an 8-bit grayscale JPEG at
`zoobot_stamps/VIS/VIS_<object_id>.jpg`. At training time the current ZooBot
dataset loader replicates grayscale to three channels and applies its standard
crop and augmentation transforms.

The output manifest records the estimated radius and WCS-derived pixel center.
Rows lacking any radius-estimation input are marked `invalid_morphology` and
omitted, matching the reference utility's final-catalog conversion behavior.
If an `R_MAX` crop does not fit inside the transferred fixed-size FITS cutout,
the row fails explicitly. Regenerate larger Datalabs cutouts for those objects;
this avoids silently changing their scale. If a catalog already contains the
pipeline `R_MAX` in pixels, select it with `--r-max-column R_MAX`.

The converter writes its own `manifest.csv` with one row per successful source
cutout and uses atomic JPEG writes. Restart the same output with `--resume` to
validate and reuse existing stamps. For a small check, use `--limit 100` and a
separate output directory.

For future samples, `euclid.bulk_sfh_cutouts` performs the MER morphology join
automatically during the normal Datalabs run. Its output directory contains
`morphology_catalog.fits` alongside `manifest.csv`, `cutouts/`, and the query
caches. The standalone `euclid.fetch_mer_morphology` command above is needed
only for cutout datasets created before this integration.

First submit a 100-object Candide pilot:

```bash
sbatch euclid/slurm_prepare_zoobot_cutouts_test.sh
```

It writes stamps to
`/n03data/huertas/euclid/sfh_clip/edfn_100k/zoobot_stamps_rmax_test100`.
Check the job log and the output `manifest.csv`, then submit the complete job:

```bash
sbatch euclid/slurm_prepare_zoobot_cutouts_100k.sh
```

Both scripts use
`/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_edfn100k/edfn_100k_cutouts`
as the cutout root. This is the directory containing `manifest.csv`; the
provided `cutouts/VIS` path is below it. They expect the matching morphology
table at
`/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_edfn100k/morphology_catalog_sfh_100k.fits`.
The first three positional arguments can override the cutout root, catalog,
and output paths. The pilot accepts a fourth argument for its sample size.

## Train Euclid image--SFH CLIP

The Euclid trainer reuses the fixed-grid ZooBot CLIP model from the COSMOS-Web
run and reads the SFH input dimension directly from the preprocessed file (250
for the current data). It joins SFHs and JPEGs by exact integer `galaxy_id`, so
catalog row order is irrelevant and objects whose cutout conversion failed are
excluded. The train/validation assignment is deterministic for a given seed and
is made before checking image availability.

The HDF5 file is opened lazily in each data-loader worker. During training, one
valid entry from `sfh_realizations` is drawn each time a galaxy is loaded. This
propagates the fitted SFH uncertainty into the contrastive training rather than
treating the posterior median as exact. Validation always uses the deterministic
median `sfh` dataset. Disable posterior sampling for an ablation with
`--no-sample-posterior`.

The default SFH encoder uses all 250 common-grid bins as attention tokens. Each
token contains log normalized SFH weight and fractional lookback time, plus a
learned positional embedding. Four self-attention layers (width 128, four
heads) and a CLS token capture both local and widely separated features before
projection into the 256-dimensional CLIP space. Use `--sfh-encoder mlp` to run
the original 197K-parameter fixed-grid MLP baseline.

First run the two-epoch, 1,024-pair GPU smoke test on candide:

```bash
sbatch euclid/slurm_train_zoobot_clip_test.sh
```

Its logs and checkpoints are written under
`/n03data/huertas/euclid/sfh_clip/edfn_100k/training_test_transformer`. This
keeps the earlier MLP smoke-test artifacts separate.

Before a full run, use the 10,000-pair optimization pilot:

```bash
sbatch euclid/slurm_train_zoobot_clip_pilot.sh
```

The pilot uses posterior-median SFHs and disables the MoCo queue. This isolates
the basic paired alignment from posterior-draw noise and from similar SFHs being
treated as thousands of queued negatives. It gives the transformer three times
the projection-head learning rate, uses two warmup epochs, and stops after 30
epochs or eight unimproved validation epochs. Its output is kept under
`training_pilot_transformer_median`.

Validation logs include rank-1 and rank-5 retrieval, loss relative to the
batch-size random baseline, positive and negative cosine similarity, and their
alignment margin. A useful model should produce negative
`val_loss_vs_random`, positive `val_alignment_margin`, and retrieval above
chance. If this pilot learns clearly, transfer its settings to the complete
matched sample:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k.sh
```

The full job starts with posterior-median SFHs, a zero-length queue, and the
pilot's 3x SFH-encoder learning rate. It trains for at most 50 epochs with two
warmup epochs and patience 10. Results are written to the new
`training_transformer_median_v2` directory so earlier weak-alignment runs remain
available for comparison. After this phase converges, posterior sampling can be
introduced in a shorter uncertainty fine-tuning phase from its best checkpoint.

The full run uses a 90/10 split, batch size 128, no MoCo queue, mixed precision,
and early stopping. It freezes the ZooBot backbone and trains the image
projection and SFH encoder. The default image backbone is
the Euclid-native
[`hf_hub:mwalmsley/zoobot-encoder-euclid`](https://huggingface.co/mwalmsley/zoobot-encoder-euclid),
loaded through timm. The model has a 640-dimensional ConvNeXt Nano output and
expects three-channel 224-pixel images, matching the prepared VIS stamps after
grayscale channel replication. The model is downloaded on its first use and
then read from `/n03data/huertas/.cache/huggingface`.

The first four positional arguments are the preprocessed HDF5 file, JPEG stamp
root, ZooBot source, and output directory. The ZooBot source may instead be a
local compatible `FinetuneableZoobotClassifier` checkpoint such as the earlier
COSMOS-Web `family_2.ckpt`. A fifth argument to the full script resumes the CLIP
Lightning checkpoint:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k.sh \
    /path/to/sfh_clip.h5 \
    /path/to/zoobot_stamps \
    hf_hub:mwalmsley/zoobot-encoder-euclid \
    /path/to/training_output \
    /path/to/training_output/checkpoints/last.ckpt
```

The best three checkpoints and `last.ckpt` are saved in
`training_transformer_median_v2/checkpoints`; CSV learning curves are saved in
`training_transformer_median_v2/logs`.
`pair_split.npz` records the exact HDF5 rows and galaxy IDs used for each split.
The trainer refuses queue and batch sizes that cannot safely update the MoCo
queue.

### Quantitative checkpoint evaluation

Before making UMAPs or inspecting selected examples, evaluate the best
checkpoint on the exact saved validation split:

```bash
sbatch euclid/slurm_evaluate_zoobot_clip.sh
```

The job measures full 9,955-object retrieval in both directions, rather than
the batch-of-128 retrieval shown during training. It also checks the saved
train/validation split and image IDs, compares paired cosine similarity with a
permutation null, diagnoses embedding collapse, tests whether retrieved SFHs
have similar raw shapes, measures stability across posterior SFH realizations,
and reports performance in redshift quartiles. It writes `metrics.json`,
`per_object.csv`, and reusable normalized validation embeddings under
`training_transformer_median_v2/evaluation_best`.

The default checkpoint is the best model from the first full median-SFH run.
Any checkpoint and output directory can be supplied positionally. For example,
compare the final state with the early-stopped best state using:

```bash
sbatch euclid/slurm_evaluate_zoobot_clip.sh \
    /n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/checkpoints/last.ckpt \
    /n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/evaluation_last
```

### UMAP embedding diagnostics

The first 100k product was created before the clean-catalog and MER morphology
tables were merged into the sampled SFH HDF5. Restore those scalar fields in
place, with an exact object-ID join, before regenerating the diagnostic PDF:

```bash
sbatch euclid/slurm_restore_morphology_metadata.sh
```

The default job reads
`sfh_edfn100k/catalog_sfh_100k.fits` and
`sfh_edfn100k/morphology_catalog_sfh_100k.fits`, stages all new HDF5 datasets
before exposing them under their final names, and leaves the SFH arrays and row
order unchanged. Existing datasets are retained. Future runs of
`sample_edfn_sfhs` copy missing numeric scalar clean-catalog columns directly
into each sampled SFH shard, so the normal preprocessor preserves them without
this repair step.

After the quantitative evaluation has written `validation_embeddings.npz` and
`per_object.csv`, generate the Euclid diagnostic atlas without re-running the
encoders:

```bash
sbatch euclid/slurm_umap_zoobot_clip.sh
```

The no-argument job is tied explicitly to:

```text
/n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt
```

It reads the embeddings previously exported for that checkpoint from
`training_transformer_median_v2/evaluation_best`. The checkpoint basename is
printed on every diagnostic section so the PDF retains its provenance.

The PDF contains independently fitted image, SFH, and normalized-average UMAPs
colored by redshift, stellar mass, Sérsic morphology, source size, point-source
probability, derived SFH shape summaries, paired cosine, and retrieval rank. A
fourth UMAP is fitted to the stacked image and SFH embeddings to show modality
occupancy and matched-pair connections in one shared projection. Its 2D
distances are treated as qualitative diagnostics; the full-dimensional
retrieval report remains the alignment measurement.

The default outputs are
`evaluation_best/euclid_clip_umap_diagnostics.pdf`, a companion `.npz` with all
four coordinate sets and properties, and a `.csv` property table. A checkpoint,
evaluation directory, and output PDF can be supplied as positional
arguments to the SLURM script. For the final checkpoint, use:

```bash
sbatch euclid/slurm_umap_zoobot_clip.sh \
    /n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/checkpoints/last.ckpt \
    /n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/evaluation_last \
    /n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/evaluation_last/euclid_clip_umap_diagnostics.pdf
```

### Partial ZooBot unfreezing experiment

Run a controlled comparison with the frozen-backbone baseline using:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k_unfreeze1.sh
```

This keeps the same deterministic split, posterior-median SFHs, transformer,
batch size, and queue-free contrastive objective. It unfreezes the final
ConvNeXt feature stage at `5e-6`, while the image projection and SFH
transformer use `1e-4`. The run writes to
`training_transformer_median_unfreeze1`, leaving the frozen baseline intact.
The feature-stage selector uses timm's `feature_info`. The terminal feature
normalization is also trainable but does not count toward `--unfreeze-blocks`.

For the completed unfreeze-one-stage run, submit quantitative evaluation and
UMAP generation together:

```bash
sbatch euclid/slurm_unfreeze1_diagnostics.sh
```

This single SLURM job uses that run's own `pair_split.npz`, evaluates the
checkpoint, and then creates the UMAP atlas if evaluation succeeds. It writes
`metrics.json`, reusable validation embeddings, the per-object retrieval table,
and `evaluation_best/euclid_clip_umap_diagnostics.pdf` under
`training_transformer_median_unfreeze1`.

### SFH-aware soft-positive experiment

The soft-positive experiment keeps the frozen Euclid ZooBot backbone, SFH
transformer, deterministic split, posterior medians, batch size, optimizer, and
queue-free training used by the exact-pair baseline. Its output directories are
separate, so it cannot overwrite `training_transformer_median_v2`.

For each batch, the code converts the stored log SFHs back to unit-normalized
linear mass weights and computes pairwise Wasserstein-1 distances on the
fractional-lookback-time grid. Each contrastive target retains 75% probability
on its exact image--SFH pair and distributes 25% among its eight closest SFHs
with an adaptive distance kernel. The target graph always uses posterior
medians; later posterior sampling can therefore perturb the SFH encoder input
without changing which histories are considered neighbours.

Verify the new loss and dataloader path with a two-epoch, 1,024-pair job:

```bash
sbatch euclid/slurm_train_zoobot_clip_soft_w1_test.sh
```

It writes to `training_test_soft_w1` and leaves the previous smoke tests intact.

Run the controlled 10,000-pair pilot first:

```bash
sbatch euclid/slurm_train_zoobot_clip_soft_w1_pilot.sh
```

It writes to `training_pilot_transformer_median_soft_w1`. `val_loss` is the
mixed soft-positive objective, while `val_exact_loss`, rank-1, rank-5, and the
alignment-margin metrics retain the exact-pair interpretation. Compare those
quantities with `training_pilot_transformer_median`, rather than comparing the
mixed `val_loss` numerically with exact InfoNCE.

If the pilot improves SFH-neighbour recovery, run all matched objects:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k_soft_w1.sh
```

The full run writes to `training_transformer_median_soft_w1`. The normal
evaluation script accepts its checkpoint, split, and a new output directory as
positional arguments. Its SFH-neighbour report includes both cosine and
Wasserstein shape distances and neighbour overlap, so the new objective can be
judged in its intended geometry alongside exact retrieval.

### SFH encoder--decoder pretraining

The SFH autoencoder keeps the four-layer, width-128 transformer used by the
alignment baseline, so its encoder weights transfer exactly. A two-layer
transformer decoder receives only the global 256-dimensional encoder output
and 250 fractional-lookback-time queries. It cannot copy local encoder tokens,
so the global representation must retain the complete SFH shape.

During pretraining, the input is a random valid posterior realization and the
target is the posterior-median SFH. One contiguous interval containing 35% of
the bins is masked. The reconstruction is constrained to be non-negative and
sum to one. Its loss gives equal weight to cumulative Wasserstein-1 distance
and a Huber term weighted by the stored 16th--84th percentile width. The exact
baseline's saved train/validation split is reused, preventing the SFH encoder
from pretraining on the later CLIP validation galaxies.

First verify the encoder, decoder, masking, and HDF5 path on 2,048 objects:

```bash
sbatch euclid/slurm_pretrain_sfh_autoencoder_test.sh
```

`val_w1` measures reconstruction of a deterministic posterior realization to
the posterior median. `val_clean_w1` measures median-to-median reconstruction
through the bottleneck. Both should decrease without becoming nonfinite. Then
run the complete pretraining:

```bash
sbatch --partition=pscomp --nodelist=n36 \
  euclid/slurm_pretrain_sfh_autoencoder.sh
```

The full run writes to `sfh_autoencoder_v1`. Copy the `Best checkpoint` path
from its output. Before starting CLIP, export the held-out validation embeddings
and fit their UMAP:

```bash
AE_CKPT='/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_autoencoder_v1/checkpoints/REPLACE_WITH_BEST.ckpt'
sbatch euclid/slurm_export_sfh_autoencoder_embeddings.sh "${AE_CKPT}"
```

The export runs on `n36` in `pscomp` and writes
`sfh_autoencoder_v1/explorer_validation` containing:

- `sfh_autoencoder_umap.npz`: UMAP coordinates, full 256-dimensional encoder
  embeddings, redshift, morphology, SFH-shape summaries, posterior width, and
  per-object reconstruction errors.
- `euclid_explorer.h5`: the median and percentile SFHs plus decoder
  reconstructions for the 9,955 held-out validation galaxies.
- `metrics.json`: reconstruction statistics, an SFH-shape-neighbourhood probe,
  and a redshift probe comparing the latent against raw-SFH and random-pair
  baselines.

Download those three files and launch the local explorer without image stamps:

```bash
conda activate astroclip-mac
python -m euclid.explore_embeddings \
  --h5 /path/to/explorer_validation/euclid_explorer.h5 \
  --umap /path/to/explorer_validation/sfh_autoencoder_umap.npz \
  --label 'SFH autoencoder validation'
```

Set one panel to `Redshift z` and the other to an SFH-shape property such as
recent mass fraction, mean lookback time, t50, or entropy. Reconstruction W1
identifies regions represented poorly by the bottleneck, and lasso selections
overlay the decoder reconstruction on the measured SFH. The cluster action uses
the full latent rather than the two-dimensional UMAP. Redshift structure is
physically expected because the SFH population evolves. In `metrics.json`,
compare latent and raw-SFH redshift R2, their neighbour delta-z values, and the
latent-neighbour SFH similarity against the random baseline. Repeat selections
within narrow redshift intervals before interpreting a trend as an encoder
artifact.

After this diagnostic looks satisfactory, verify that the checkpoint can
initialize the CLIP model:

```bash
sbatch euclid/slurm_train_zoobot_clip_sfh_autoencoder_test.sh "${AE_CKPT}"
```

After the two-epoch integration test succeeds, train the controlled 100k
comparison:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k_sfh_autoencoder.sh "${AE_CKPT}"
```

This run writes to `training_transformer_median_sfh_autoencoder`. It retains
the frozen Euclid ZooBot backbone, posterior-median inputs, exact-pair
contrastive loss, optimizer, and split from the baseline. The only changes are
SFH encoder initialization and a reconstruction regularizer with weight 0.1.
`val_contrastive_loss` remains directly comparable with the baseline, while
`val_loss` includes the auxiliary reconstruction term. Evaluate the resulting
best checkpoint with `slurm_evaluate_zoobot_clip.sh` on the common baseline
split before comparing its retrieval and SFH-neighbour metrics. For a
decoder-enabled checkpoint, that evaluation also adds median reconstruction
W1 and mean-absolute-error summaries to `metrics.json`.

#### Frozen SFH autoencoder with a linear CLIP projection

Full fine-tuning can return the pretrained SFH transformer to the same optimum
as a randomly initialized exact-pair model. The frozen-projection experiment
keeps the pretrained SFH encoder and decoder fixed and inserts a bias-free
256-by-256 linear projection before the contrastive loss. That projection is
initialized to the identity. The ZooBot backbone also remains frozen, so the
trainable cross-modal maps are the existing image projection and the new SFH
projection; the autoencoder representation and reconstruction cannot drift.

Run the three-epoch integration test first:

```bash
AE_CKPT='/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_autoencoder_v1/checkpoints/euclid_sfh_autoencoder_v1-epoch=001-val_loss=0.02222.ckpt'
sbatch euclid/slurm_train_zoobot_clip_sfh_frozen_projection_test.sh "${AE_CKPT}"
```

Then train all matched objects:

```bash
sbatch euclid/slurm_train_zoobot_clip_100k_sfh_frozen_projection.sh "${AE_CKPT}"
```

The full run writes to `training_sfh_frozen_linear_projection`. Its evaluation
archive stores both `sfh_embedding` (after the trainable linear projection) and
`sfh_preprojection_embedding` (the unchanged autoencoder latent). Evaluation
also reports the overlap of their nearest-neighbour graphs and the correlation
of their pairwise cosine similarities. The UMAP diagnostic and interactive
explorer expose `SFH encoder` and `SFH autoencoder latent` as separate spaces.
This distinguishes successful image alignment from destruction of the original
SFH geometry.

#### Frozen SFH autoencoder with a residual MLP projection

If the linear projection underfits, the residual-MLP experiment adds nonlinear
alignment capacity without changing the pretrained autoencoder. Its projection
is `LayerNorm -> 256 -> 512 -> GELU -> 256`, followed by a residual connection
with fixed scale 0.1. The last layer starts at zero, so the projected embedding
is exactly equal to the autoencoder latent before the first optimizer step.

Run the integration test and then the complete sample in separate output
directories:

```bash
AE_CKPT='/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_autoencoder_v1/checkpoints/euclid_sfh_autoencoder_v1-epoch=001-val_loss=0.02222.ckpt'

sbatch euclid/slurm_train_zoobot_clip_sfh_frozen_mlp_projection_test.sh \
  "${AE_CKPT}"

sbatch euclid/slurm_train_zoobot_clip_100k_sfh_frozen_mlp_projection.sh \
  "${AE_CKPT}"
```

The full output is `training_sfh_frozen_residual_mlp_projection`. Evaluate it
on the baseline common split. The same pre/post-projection geometry diagnostics
and explorer fields used by the linear run allow the autoencoder, fine-tuned,
linear-projection, and residual-MLP representations to be compared directly.

#### Bright VIS controlled sample

The full SFH sample extends well below the magnitude range where resolved VIS
morphology is reliable. A controlled bright-sample experiment filters the
existing HDF5 and JPEG pairs without making another SFH catalog or image set.
`FLUX_DETECTION_TOTAL` is interpreted as microJy only for `VIS_DET=1`, using
`VIS_AB = 23.9 - 2.5 log10(flux_microJy)`. The original seed-42 validation
membership is retained before the magnitude cut, preventing objects from moving
between train and validation when comparing magnitude limits. These photometry
columns are read from `sfh_edfn100k/catalog_sfh_100k.fits` and joined exactly to
the preprocessed HDF5 by `OBJECT_ID`; the HDF5 itself is not modified.

First count the available pairs at several limits and write the VIS<22 ID table:

```bash
sbatch euclid/slurm_summarize_vis_bright_sample.sh 22.0
```

The output reports counts at 20.5, 21, 21.5, 22, and 22.5. Run the controlled
baseline architecture at VIS<22, then optionally repeat at VIS<21 after checking
that enough pairs remain:

```bash
sbatch euclid/slurm_train_zoobot_clip_bright_test.sh 22.0
sbatch euclid/slurm_train_zoobot_clip_bright.sh 22.0

# Stricter comparison
sbatch euclid/slurm_train_zoobot_clip_bright.sh 21.0
```

Outputs are separated as `training_transformer_median_vislt22p0` and
`training_transformer_median_vislt21p0`. Retrieval sets have different sizes,
so compare rank percentiles, paired-minus-shuffled cosine, and SFH-neighbour
quality in addition to raw recall at K.

For a controlled test, evaluate both the new bright model and the original
full-sample model on the bright run's saved validation split. Substitute the
actual best bright checkpoint printed by the training job:

```bash
BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k
BRIGHT=${BASE}/training_transformer_median_vislt22p0
BRIGHT_CKPT=${BRIGHT}/checkpoints/<best-bright-checkpoint>.ckpt
BASELINE_CKPT=${BASE}/training_transformer_median_v2/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt

sbatch -p pscomp -w n36 euclid/slurm_evaluate_zoobot_clip.sh \
  "${BRIGHT_CKPT}" "${BRIGHT}/evaluation_best" "${BRIGHT}/pair_split.npz"
sbatch -p pscomp -w n36 euclid/slurm_evaluate_zoobot_clip.sh \
  "${BASELINE_CKPT}" "${BRIGHT}/evaluation_baseline_on_bright" \
  "${BRIGHT}/pair_split.npz"
```

This separates the effect of training on brighter galaxies from the simpler
effect that a bright validation set has clearer images. The cut will also shift
the redshift and stellar-mass distributions, so inspect those properties before
attributing an improvement specifically to morphology.

The census job also writes `bright_samples/vis_lt_22p0.fits`. Transfer this
small ID table to ESA Datalabs and query the official MER morphology table from
the repository root in the EUCLID-TOOLS environment:

```bash
/opt/miniforge/envs/euclid-tools/bin/python -m euclid.fetch_mer_zoobot_morphology \
  --sample /path/to/vis_lt_22p0.fits \
  --output /path/to/mer_zoobot_vis_lt_22p0.fits \
  --resume
```

In the authenticated IDR schema this queries
`catalogue.mer_morphology_deep_survey`, matching the EDFN deep-survey sample.

Transfer that FITS table back to Candide. After all jobs reading the HDF5 have
finished, add the bright-subset metadata without changing the SFHs or row order:

```bash
python -m euclid.restore_morphology_metadata \
  --dataset /n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_clip_100k.h5 \
  --catalog /path/to/mer_zoobot_vis_lt_22p0.fits \
  --allow-missing
```

Subsequent UMAP archives and explorer bundles include VIS magnitude; CAS,
Gini, and M20; MER T-type and major-merger scores; and normalized ZooBot
smooth, featured, edge-on, spiral, bar, and disturbed/merger probabilities.
The ZooBot values in MER are Dirichlet concentrations, so the diagnostic code
normalizes answers within each morphology question before plotting them.

If the original random 100k sample contains too few bright objects, count the
entire EDFN clean catalog after matching it to the `EDFN_2fwhm_aper.h5` SFHs:

```bash
sbatch euclid/slurm_census_edfn_bright_sfhs.sh
```

This reports both the clean-catalog count and the SFH-matched count at VIS limits
20.5, 21, 21.5, 22, 22.5, 23, and 23.5 without writing a sample. Choose the
brightest limit that provides enough training objects, then sample from that
restricted population. For example, after confirming that 50,000 objects are
available at VIS<22.5:

```bash
sbatch euclid/slurm_sample_edfn_bright_sfhs.sh 22.5 50000
```

The output would be
`/n03data/huertas/euclid/sfh_clip/edfn_vislt22p5_50000`. The sampler applies the
VIS cut before random selection, requires `VIS_DET=1`, uses only the matching
2-FWHM SFH catalog, and records the selection in the FITS metadata. Use a third
argument to choose another new output directory. This new sample needs its own
SFH preprocessing, Datalabs cutout extraction, ZooBot JPEG conversion, and
train/validation split; do not mix its products with the original random 100k
directory.

### Interactive Euclid embedding explorer

`euclid.explore_embeddings` adapts the COSMOS-Web Panel application to the
Euclid data layout. The two linked UMAP panels can display different runs and
embedding spaces while preserving selections by galaxy ID. Selected objects
show their VIS stamps, median SFHs with 16th--84th percentile posterior bands,
and the population SFH of the complete selection. Redshift and scalar-property
filters are available, and selections can be saved to CSV.

The app clusters in the full 256-dimensional image, SFH, or joint latent space
when those arrays are present. UMAP coordinates are used for navigation and
display. New diagnostic NPZ files include the raw embeddings automatically;
the bundle exporter also merges them from the sibling
`validation_embeddings.npz` for diagnostic files made with older code.

The main comparison now uses the complete 100k exact-pair and soft-positive
runs. Evaluate both best checkpoints against the exact-pair run's saved split,
so every metric and plotted point refers to the same 9,955 validation galaxies:

```bash
BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k
COMMON_SPLIT=${BASE}/training_transformer_median_v2/pair_split.npz

sbatch euclid/slurm_evaluate_zoobot_clip.sh \
  "${BASE}/training_transformer_median_v2/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt" \
  "${BASE}/training_transformer_median_v2/evaluation_best_common_split" \
  "${COMMON_SPLIT}"

sbatch euclid/slurm_evaluate_zoobot_clip.sh \
  "${BASE}/training_transformer_median_soft_w1/checkpoints/euclid_vis_sfh_transformer_median_soft_w1_100k-epoch=030-val_loss=4.5688.ckpt" \
  "${BASE}/training_transformer_median_soft_w1/evaluation_best_common_split" \
  "${COMMON_SPLIT}"
```

After both evaluation jobs finish, create their diagnostic NPZ files:

```bash
BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k

sbatch euclid/slurm_umap_zoobot_clip.sh \
  "${BASE}/training_transformer_median_v2/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt" \
  "${BASE}/training_transformer_median_v2/evaluation_best_common_split" \
  "${BASE}/training_transformer_median_v2/evaluation_best_common_split/euclid_clip_umap_diagnostics.pdf"

sbatch euclid/slurm_umap_zoobot_clip.sh \
  "${BASE}/training_transformer_median_soft_w1/checkpoints/euclid_vis_sfh_transformer_median_soft_w1_100k-epoch=030-val_loss=4.5688.ckpt" \
  "${BASE}/training_transformer_median_soft_w1/evaluation_best_common_split" \
  "${BASE}/training_transformer_median_soft_w1/evaluation_best_common_split/euclid_clip_umap_diagnostics.pdf"
```

After both UMAP jobs finish, create a compact transfer bundle. It extracts the
median and percentile SFHs and packages the matched VIS stamps:

```bash
BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k

sbatch euclid/slurm_export_explorer_bundle.sh \
  "${BASE}/explorer_100k_exact_vs_soft_w1" \
  "${BASE}/training_transformer_median_v2/evaluation_best_common_split/euclid_clip_umap_diagnostics.npz" \
  "${BASE}/training_transformer_median_soft_w1/evaluation_best_common_split/euclid_clip_umap_diagnostics.npz"
```

Download the resulting `explorer_100k_exact_vs_soft_w1` directory. It contains all
files needed locally:

- `euclid_explorer.h5`: compact SFH medians, posterior intervals, and time grid.
- Two diagnostic `.npz` files: UMAP coordinates, physical properties, alignment
  metrics, and raw embeddings for the exact and soft-positive runs.
- `VIS_stamps.tar`: only the JPEGs for the common validation galaxies.
- `manifest.json`: provenance and file inventory.

The checkpoints, full 100k HDF5 file, PDF diagnostics, and separate evaluation
CSV files are not needed on the laptop. After downloading, extract the stamps
and install the small local viewer environment:

```bash
tar -xf explorer_100k_exact_vs_soft_w1/VIS_stamps.tar \
  -C explorer_100k_exact_vs_soft_w1
python3 -m venv .venv-euclid-explorer
source .venv-euclid-explorer/bin/activate
python -m pip install -r euclid/requirements_explorer.txt
```

Launch the linked comparison:

```bash
python -m euclid.explore_embeddings \
  --h5 explorer_100k_exact_vs_soft_w1/euclid_explorer.h5 \
  --stamps explorer_100k_exact_vs_soft_w1/VIS \
  --umap explorer_100k_exact_vs_soft_w1/00_training_transformer_median_v2_euclid_clip_umap_diagnostics.npz \
  --umap explorer_100k_exact_vs_soft_w1/01_training_transformer_median_soft_w1_euclid_clip_umap_diagnostics.npz \
  --label 'exact-pair 100k' \
  --label 'soft-W1 100k'
```
