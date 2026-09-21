"""Draw reproducible galaxy-ID subsets for a shared-noise diffusion comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def select_clusters(paths, output, n=6, seed=42):
    output=Path(output)
    if n<1: raise ValueError('n must be positive')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    rng=np.random.default_rng(seed)
    chosen=[]; seen=set(); provenance=[]
    for path in map(Path,paths):
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            key=next((k for k in ('galaxy_id','object_id') if k in (reader.fieldnames or [])),None)
            if key is None: raise ValueError(f'No ID column in {path}')
            ids=[int(row[key]) for row in reader]
        if len(set(ids))!=len(ids): raise ValueError(f'Duplicate IDs in {path}')
        if seen.intersection(ids): raise ValueError('Cluster input lists overlap; resolve membership before sampling.')
        seen.update(ids)
        if len(ids)<n: raise ValueError(f'{path} has fewer than {n} objects')
        label=path.stem.removeprefix('euclid_selected_galaxies_')
        sample=rng.choice(np.array(sorted(ids),dtype=np.int64),n,replace=False)
        chosen.extend(dict(galaxy_id=int(gid),cluster=label) for gid in sample)
        provenance.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               cluster=label,n_available=len(ids),selected_ids=list(map(int,sample))))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['galaxy_id','cluster']);writer.writeheader();writer.writerows(chosen)
    output.with_suffix('.json').write_text(json.dumps(dict(selection_seed=seed,n_per_cluster=n,clusters=provenance),indent=2))
    print(f'Saved {len(chosen)} IDs to {output}')
    for p in provenance: print(f"{p['cluster']}: {n}/{p['n_available']}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selections',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n-per-cluster',type=int,default=6)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();select_clusters(a.selections,a.output,a.n_per_cluster,a.seed)

if __name__=='__main__': main()
