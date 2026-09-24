# Three-cluster comparison with 100 Myr recent activity

Same deterministic KMeans clusters as ../cluster_recent_ms; selection and
plots now use the last 100 Myr rather than 0.1 cosmic age.

Recent ΔMS = log10(∫ SFR dt / ∫ shifted MS SFR dt), R=0, mass offset=0,
empirical MS SFR shift=-0.93 dex. The shift multiplies the reference MS SFR
by 10**(-0.93), equivalently adding +0.93 dex to the unshifted ΔMS.
Partial native bins contribute proportionally to the 100 Myr window.
No additional shift is applied to the SFH or catalog SFR itself.

cluster_distributions.pdf compares each cluster to the full local validation
sample. cluster_sfh_cutout_galleries.pdf shows four random examples per
cluster and activity interval (<-0.5, -0.5 to +0.5, >+0.5 dex). Gallery
shading marks 100 Myr. Values at or below -4 dex are censored to -4.
CSV membership files retain all three windows as separately named columns;
gallery_selected_ids.csv explicitly records the window used for selection.
No mass/redshift matching or quality cuts are applied. Earlier 0.1-cosmic-age
results remain in ../cluster_recent_ms.
