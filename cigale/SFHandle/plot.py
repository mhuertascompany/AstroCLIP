#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Plotting utilities for SFHs.
'''

import matplotlib.pyplot as plt

# Local imports
from .sfh import SFH

def plot_sfh(sfh : SFH) -> None: # pragma: no cover
    r'''
    Plot a SFH and its interpolated form if available.

    :param sfh: star formation history to show
    :type sfh: :py:class:`~.SFH`

    **Example**

    .. jupyter-execute::

        import numpy             as np
        import matplotlib.pyplot as plt

        from SFHandle import SFH
        from SFHandle import plot_sfh
        
        lb_time   = [1, 10, 100, 1000]
        amp       = [0, 1, 10, 1]
        err       = [0, 0.5, 3, 1]

        mysfh     = SFH(lb_time, amp, err)
        
        lb_interp = np.linspace(0, 1000, 1000)
        mysfh.interpolate_sfh(lb_interp)
        
        _ = plot_sfh(mysfh)
    '''

    if (sfh.interp_lb_time is not None and
        sfh.interp_sfh     is not None and
        sfh.interp_err     is not None):

        # Interpolated SFH
        plt.step(
            sfh.interp_lb_time,  # pyright: ignore[reportArgumentType]
            sfh.interp_sfh,  # pyright: ignore[reportArgumentType]
            color = 'firebrick', 
            ls    = '--'
        )

        plt.scatter(
            sfh.interp_lb_time,  # pyright: ignore[reportArgumentType]
            sfh.interp_sfh,   # pyright: ignore[reportArgumentType]
            c      = 'firebrick', 
            marker = 'x'
        )

        plt.fill_between(
            sfh.interp_lb_time, # pyright: ignore[reportArgumentType]
            sfh.interp_sfh - sfh.interp_err, # type: ignore
            sfh.interp_sfh + sfh.interp_err, # type: ignore
            edgecolor = 'firebrick', 
            facecolor = 'none',
            hatch     = '/',
            alpha     = 0.3,
            step      = 'pre'
        )

    # Original SFH
    plt.step(sfh.lb_time, sfh.sfh, color='k', where='pre')
    plt.scatter(sfh.lb_time, sfh.sfh, c='k', marker='o')
    
    plt.fill_between(
        sfh.lb_time,
        sfh.sfh - sfh.err,
        sfh.sfh + sfh.err,
        color = 'k',
        alpha = 0.3,
        step  = 'pre'
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.ylabel(r'SFH [M$_{\odot}$ yr$^{-1}$]')
    plt.xlabel(r'Lookback time [Myr]')

    return