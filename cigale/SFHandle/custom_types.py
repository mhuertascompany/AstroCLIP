from   typing import Literal, TypeAlias
import numpy  as     np

#: Allowed values for interpolation with `interp1d`_.
Interp_kind: TypeAlias = Literal[
    'linear', 'nearest', 'nearest-up', 
    'zero', 'slinear', 'quadratic', 
    'cubic', 'previous', 'next'
]

#: All floating types handled by numpy
Numpy_float: TypeAlias = np.float16 | np.float32 | np.float64 | np.float128