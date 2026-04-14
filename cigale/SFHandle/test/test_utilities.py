#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

Test suite for the utilities.py module.
'''

import astropy.table as tab

# Pytest loads the top directory as package thanks to the empty __init__ file. Imports are ok for testing.
from SFHandle import utilities # pyright: ignore[reportMissingImports]

class Test_cigale_sfh:
    r'''
    .. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@lam.fr>

    Test suite for the cigale_sfh function.
    '''

    def load_test_data(self) -> tab.Table:
        r'''Load a test data set.'''

        return tab.Table.read('test/test.csv')

    def test_sfh_creation(self):
        r'''Test the creation of sfh data.'''

        data = self.load_test_data()
        _    = utilities.cigale_sfh(data[0]) # pyright: ignore[reportArgumentType]
        
        assert True

    def test_shape_output_1(self):
        r'''Verify that the time and sfh arrays have similar shapes.'''

        data         = self.load_test_data()
        time, sfh, _ = utilities.cigale_sfh(data[0]) # pyright: ignore[reportArgumentType]

        assert time.shape == sfh.shape

    def test_shape_output_2(self):
        r'''Veirify that the time and sfh error arrays have similar shapes.'''
        
        data             = self.load_test_data()
        time, _, sfh_err = utilities.cigale_sfh(data[0]) # pyright: ignore[reportArgumentType]

        assert time.shape == sfh_err.shape