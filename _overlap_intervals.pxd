# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef DTYPE_t[:] DTYPE_t_1D
ctypedef DTYPE_t[:,:] DTYPE_t_2D

cdef void _get_overlaps(DTYPE_t_2D intervals,
                              np.int64_t n_classes,
                                DTYPE_t_1D consensus) nogil