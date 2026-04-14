#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test suite for the sfh.py module.
'''
import pytest
import astropy.table as     tab
import numpy         as     np

# Pytest loads the top directory as package thanks to the empty __init__ file. Imports are ok for testing.
from SFHandle import sfh # pyright: ignore[reportMissingImports]
from SFHandle import utilities # pyright: ignore[reportMissingImports]

# Setting up test data from the catalogue
test_data = tab.Table.read('test/test.csv')
test_sfhs = [
    (
        utilities.cigale_sfh(data), 
        data['bayes.sfh.integrated']
    )
    for data in test_data
]

class Test_SFH:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Test suite for the SFH class.
    '''
    
    def test_constructor(self) -> None:
        r'''Test the creation of sfh data.'''
        
        # Select the SFH of the first element of the test data
        data = test_sfhs[0][0]

        sfh.SFH(data[0], data[1], data[2])

        assert True

    def test__interpolate(self) -> None:
        r'''Test the interpolation of a step function.'''

        mysfh = sfh.SFH([], [], [])
        
        res   = mysfh._interpolate(
            [0, 0.5, 1],
            [0, 1],
            [0, 1],
            kind = 'next'
        )

        assert np.all(res == [0, 1, 1])

    def test_interpolate_sfh(self) -> None:
        r'''Test the interpolation of a simple SFH using the default parameters.'''

        mysfh = sfh.SFH([0, 1], [1, 2], [2, 3])
        res   = mysfh.interpolate_sfh([0, 0.5, 1, 1.5])

        assert (np.all(res[0] == [1, 2, 2, 0]) 
                and
                np.all(res[1] == [2, 3, 3, 0])
               )

    def test_interpolate_sfh_with_nan(self) -> None:
        r'''Test the interpolation of a simple SFH asking to extrapolate to NaN.'''

        mysfh = sfh.SFH([0, 1], [1, 2], [2, 3])
        res   = mysfh.interpolate_sfh(
            [0, 0.5, 1, 1.5],
            fill_value = np.nan
        )

        assert np.all(np.isnan([res[0][-1], res[1][-1]]))

    def test_interp_lb_time(self) -> None:

        mysfh = sfh.SFH(
            [0, 1, 10], 
            [1, 2, 3], 
            [2, 3, 4]
        )

        mysfh.interpolate_sfh((x := [0, 0.5, 1, 5, 10, 12]))
        
        assert np.all(mysfh.interp_lb_time == np.array(x))

    #########################################
    #        Testing integral of SFH        #
    #########################################

    def test_integral_from_manual(self) -> None:
        r'''Test the integrated SFH using a manually made SFH.'''

        mysfh = sfh.SFH(
            [0, 1, 10], 
            [1, 2, 3], 
            [2, 3, 4]
        )

        assert mysfh.integral == (3*9 + 2) * 1e6

    @pytest.mark.parametrize('data, integrated', test_sfhs)
    def test_integral_from_catalogue(self, data, integrated) -> None:
        r'''
        Test the integrated SFH using entries from the test catalogue.

        We require the precision on the integral to be below 2%.
        '''
        
        mysfh = sfh.SFH(data[0], data[1], data[2])

        assert np.abs(mysfh.integral / integrated - 1) < 0.02

    ###################################################################
    #        Testing the integration of the SFH at any lb time        #
    ###################################################################

    def test_integrated_sfh_at_0(self) -> None:
        r'''Test the integration method when integrated throughout the entire star formation history.'''

        mysfh = sfh.SFH(
            [0, 1, 10], 
            [1, 2, 3], 
            [2, 3, 4]
        )

        assert mysfh.integrated_sfh_at(0.0) == mysfh.integral

    def test_integrated_sfh_at_before_birth(self) -> None:
        r'''Test the integration method when integrated before the galaxy was even born.'''

        mysfh = sfh.SFH(
            [0, 1, 10], 
            [1, 2, 3], 
            [2, 3, 4]
        )

        assert mysfh.integrated_sfh_at(10.1) == 0.0

    def test_integrated_sfh_at_increasing(self) -> None:
        r'''Test the integration method at different look-back times and verify that the mass increases with younger times.'''

        mysfh = sfh.SFH(
            [0, 1, 10], 
            [1, 2, 3], 
            [2, 3, 4]
        )

        lb_times    = [0, 0.5, 1, 2, 4, 6, 8, 10]
        integration = np.array([mysfh.integrated_sfh_at(lb) for lb in lb_times])

        assert np.all(integration[:-1] > integration[1:])