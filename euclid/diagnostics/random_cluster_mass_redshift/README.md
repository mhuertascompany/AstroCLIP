# Random SFH and observed-image examples

Existing cluster memberships are retained. Selection uses PHZ log stellar
mass, explorer redshift, and availability of VIS stamps only. No sSFR or
recent-SFH cuts are used. Within each half-open mass/redshift bin, up to three
objects per cluster are drawn uniformly without replacement using seed 42.
Rows correspond to clusters; SFH y limits are shared across each page.
The dotted line is 100 Myr before observation. Shading shows stored pointwise
SFH posterior intervals.

Mass bins: [9,9.5), [9.5,10), [10,10.5), [10.5,11), [11,11.5).
Redshift bins: [0,0.3), [0.3,0.6), [0.6,1), [1,1.5).
Completely empty bins are omitted; missing clusters within a populated bin
are shown explicitly. Objects outside these ranges are not displayed.

See bin_counts.csv for populations and page lookup, selected_ids.csv for all
examples, and manifest.json for the totals. These are random broad-bin
comparisons, not one-to-one matching or a morphology-quality-selected sample.
