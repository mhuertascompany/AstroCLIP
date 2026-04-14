#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Load and show an example of a SFH.
'''

import os
import sys
import astropy.table as tab
import numpy         as np

sys.path.append(os.path.abspath('..'))

# Local imports
import utilities
from   plot        import plot_sfh
from   sfh         import SFH


def main() -> None:

    # Load data
    res = utilities.cigale_sfh(tab.Table.read('test.csv')[0])
    sfh = SFH(res[0], res[1], res[2])

    print(res)

    # Interpolate SFH on a new look-back time grid
    lb_grid = np.geomspace(1, 1.4e4, 100)
    sfh.interpolate_sfh(lb_grid)
    
    return plot_sfh(sfh)

if __name__ == '__main__': main()