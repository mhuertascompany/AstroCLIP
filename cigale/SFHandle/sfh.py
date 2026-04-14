#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Main part of the module that defines a SFH object to be manipulated.
'''

import numpy             as     np
from   numpy.typing      import NDArray, ArrayLike
from   scipy.interpolate import interp1d

from   .custom_types     import Interp_kind, Numpy_float

class SFH:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Class representing a SFH from Cigale.

    :param lb_time: Look-back time steps in :math:`{\rm Myr}` for the SFH.
    :type lb_time: `ArrayLike`_
    :param sfh: Amplitude of the SFH at each time step in :math:`{\rm M}_\odot~{\rm yr}^{-1}`.
    :type sfh: `ArrayLike`_
    :param err: Error on the amplitude of the SFH at each time step :math:`{\rm M}_\odot~{\rm yr}^{-1}`.
    :type err: `ArrayLike`_
    '''

    def __init__(
            self, 
            lb_time : ArrayLike,
            sfh     : ArrayLike,
            err     : ArrayLike
        ) -> None:

        #: Look-back time steps in :math:`{\rm Myr}` for the SFH
        self.lb_time = np.array(lb_time)

        #: Amplitude of the SFH at each time step in :math:`{\rm M}_\odot~{\rm yr}^{-1}`
        self.sfh     = np.array(sfh)

        #: Error on the amplitude of the SFH at each time step in :math:`{\rm M}_\odot~{\rm yr}^{-1}`
        self.err     = np.array(err)

        # Interpolated values. They are set to an array when interpolate_sfh is called.
        self._interp_lb_time : NDArray | None = None
        self._interp_sfh     : NDArray | None = None
        self._interp_err     : NDArray | None = None

        return
    
    ##########################
    #       Properties       #
    ##########################
    
    @property
    def interp_lb_time(self) -> NDArray | None: 
        r'''High-resolution time array in :math:`{\rm Myr}` used for SFH interpolation.'''
        
        return self._interp_lb_time
    
    @property
    def interp_sfh(self) -> NDArray | None: 
        r'''High-resolution interpolated SFH in :math:`{\rm M}_\odot~{\rm yr}^{-1}`.'''
        
        return self._interp_sfh

    @property
    def interp_err(self) -> NDArray | None: 
        r'''High-resolution interpolated error on the SFH amplitude :math:`{\rm M}_\odot~{\rm yr}^{-1}`.'''
        
        return self._interp_err

    @property
    def integral(self) -> Numpy_float: 
        r'''Integral of the SFH in :math:`{\rm M}_\odot`.'''
        
        # Duration of each bin in Myr
        tstep = (self.lb_time[::-1][:-1] - self.lb_time[::-1][1:])
        
        return np.nansum(self.sfh[::-1][:-1] * tstep) * 1e6

    ########################################
    #            Public methods            #
    ########################################

    def interpolate_sfh(self, 
            lb_time      : ArrayLike,
            kind         : Interp_kind = 'next',
            bounds_error : bool        = False,
            fill_value   : float       = 0.0,
            **kwargs
        ) -> tuple[NDArray, NDArray]:
        r'''
        Interpolate the SFH and its uncertainty on a new look-back time grid.
        
        .. important::

            By default ``kind`` is equal to :python:`'next'`. This is the correct interpolation type for non-parametric SFHs. Change this parameter only if you know what you are doing.

            For details regarding the parameters used for the interpolation, see `interp1d`_.

        :param lb_time: Look-back time array in :math:`{\rm Myr}` onto which the SFH must be interpolated.
        :type lb_time: `ArrayLike`_
        :param kind: Type of interpolation.
        :param bool bounds_error: Whether to throw an error if extrapolating or not.
        :param float fill_value: Value used for extrapolation.

        :returns: Interpolated SFH and interpolated error on SFH amplitude. Both are in :math:`{\rm M}_\odot~{\rm yr}^{-1}`.
        :rtype: (`NDArray`_, `NDArray`_)
        '''

        # Store the high-resolution range of look-back time
        self._interp_lb_time = np.array(lb_time)

        # Interpolate SFH
        self._interp_sfh = self._interpolate(
            lb_time, self.lb_time, self.sfh,
            kind         = kind, 
            bounds_error = bounds_error, 
            fill_value   = fill_value,
            **kwargs
        )

        # Interpolate the error on the SFH
        self._interp_err = self._interpolate(
            lb_time, self.lb_time, self.err,
            kind         = kind, 
            bounds_error = bounds_error, 
            fill_value   = fill_value,
            **kwargs
        )

        return self.interp_sfh, self.interp_err # pyright: ignore[reportReturnType]
    
    def integrated_sfh_at(self, lb_time: float) -> float:
        r'''
        Compute the integral of the SFH from the birth of the galaxy to the given time.

        :param float lb_time: lookback time at which to evaluate the integral
        '''

        # Mask that selects time bins older than the look-back time
        mask  = self.lb_time >= lb_time

        # If the look-back time is older than the galaxy, the mass is 0
        if not np.any(mask): return 0.0

        # Duration of each bin in Myr
        tstep = self.lb_time[mask][1:] - self.lb_time[mask][:-1]

        # If the look-back time value falls in between two bin edges,
        # We need to compute the mass generated during that event
        tstep = np.append(self.lb_time[mask][0] - lb_time, tstep)
        
        return np.nansum(self.sfh[mask] * tstep) * 1e6
    


    #################################
    #        Private methods        #
    #################################

    @classmethod
    def _interpolate(cls,
            new_x        : ArrayLike,
            old_x        : ArrayLike,
            old_y        : ArrayLike,
            **kwargs
        ) -> NDArray:
        r'''
        Class method implementing a useful 1D interpolation using scipy.
        
        .. note:

            Interpolation is done on a function of the kind :math:`y = f(x)` using `interp1d`_.

        :param new_x: New array used for interpolation.
        :type new_x: `ArrayLike`_
        :param old_x: Old x array defining the function.
        :type old_x: `ArrayLike`_
        :param old_y: Old y array defining the function.
        :type old_x: `ArrayLike`_
        :param kwargs: Additional keyword arguments passed to `interp1d`_.

        :returns: Interpolated function evaluated on `new_x`.
        :rtype: `ArrayLike`_
        '''

        return interp1d(old_x, old_y, **kwargs)(new_x)