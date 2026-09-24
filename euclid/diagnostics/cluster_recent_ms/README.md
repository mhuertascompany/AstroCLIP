# Three-cluster recent-activity test

KMeans on the stored 256-dimensional aligned SFH embeddings, k=3, seed=42,
n_init=10. IDs are joined exactly to the local HDF5 histories. Cluster labels
are ordered by increasing median mean fractional lookback time; they are not
assigned morphology classes and need not reproduce hand-selected clusters.

Recent ΔMS uses the last 0.1 cosmic age and log10(integrated SFR / integrated
MS SFR), R=0, mass offset=0, and empirical MS shift=-0.93 dex. Values ≤-4 dex
are censored to -4, including zero SFR. All three recent windows are exported
in cluster_membership.csv and the separate cluster_N_ids.csv files.

cluster_distributions.pdf compares unit-area distributions with the entire
13,717-object local validation sample. cluster_sfh_cutout_galleries.pdf has
one page per cluster and four reproducibly random examples from each of
three shared intervals: ΔMS<-0.5, -0.5≤ΔMS≤0.5, and ΔMS>0.5. Selected IDs and
properties are in gallery_selected_ids.csv. Images are observed VIS stamps,
not diffusion generations. Linear SFH plots include pointwise uncertainty
and the shifted MS, including its extrapolated portions.

No mass/redshift matching or quality cuts were imposed. Rare tails can have
very different mass/redshift distributions; this exploratory plot alone does
not isolate historical SFH effects on morphology. See manifest.json for
counts, settings, and source paths.
