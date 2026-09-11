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
