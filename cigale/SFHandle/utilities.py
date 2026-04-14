#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Set of utility functions related to SFHs.
'''

import numpy         as     np
import astropy.table as     tab

def cigale_sfh(galaxy : tab.Row):
    r'''
    .. codeauthor:: Rafael Arango-Torro - LAM

    Reconstruct the SFH of a given galaxy.

    :param galaxy: Data table entry for a single galaxy.
    :type galaxy: `Row`_

    :returns: 
        - Lookback time bins in :math:`\rm Myr`.
        - SFH amplitude per time bin in :math:`{\rm M}_\odot~{\rm yr}^{-1}`.
        - Error on SFH amplitude per time bin in :math:`{\rm M}_\odot~{\rm yr}^{-1}`.

    :rtype: (`NDArray`_, `NDArray`_, `NDArray`_)
    '''
    
    # define the SFH column names
    yy     = ('bayes.sfh.sfr_bin1', 'bayes.sfh.sfr_bin2', 'bayes.sfh.sfr_bin3', 'bayes.sfh.sfr_bin4', 'bayes.sfh.sfr_bin5', 'bayes.sfh.sfr_bin6', 'bayes.sfh.sfr_bin7', 'bayes.sfh.sfr_bin8', 'bayes.sfh.sfr_bin9')
    yy_err = ('bayes.sfh.sfr_bin1_err', 'bayes.sfh.sfr_bin2_err', 'bayes.sfh.sfr_bin3_err', 'bayes.sfh.sfr_bin4_err', 'bayes.sfh.sfr_bin5_err', 'bayes.sfh.sfr_bin6_err', 'bayes.sfh.sfr_bin7_err', 'bayes.sfh.sfr_bin8_err', 'bayes.sfh.sfr_bin9_err')
    xx     = ('bayes.sfh.time_bin1', 'bayes.sfh.time_bin2', 'bayes.sfh.time_bin3', 'bayes.sfh.time_bin4', 'bayes.sfh.time_bin5', 'bayes.sfh.time_bin6', 'bayes.sfh.time_bin7', 'bayes.sfh.time_bin8', 'bayes.sfh.time_bin9')

    x      = np.array([0] + [galaxy[xx_] for xx_ in xx])
    y      = np.array([galaxy[yy[0]]]     + [galaxy[xx_] for xx_ in yy])     * galaxy['bayes.sfh.integrated']
    y_err  = np.array([galaxy[yy_err[0]]] + [galaxy[xx_] for xx_ in yy_err]) * galaxy['bayes.sfh.integrated']
    
    return x, y, y_err