import numpy as np
cimport numpy as np

# python setup_computeind.py build_ext --inplace
# cython -a computeind.pyx

DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef unsigned int ITYPE_t
cimport cython
@cython.boundscheck(False)
def compute_ind(np.ndarray[DTYPE_t, ndim=1] nums,
                ITYPE_t map_dofs1,
                ITYPE_t lins1,
                DTYPE_t layers1,
                np.ndarray[DTYPE_t, ndim=1] mesh2d,
                np.ndarray[DTYPE_t, ndim=2] dofs not None,
                A not None,
                ITYPE_t wedges1,
                mapp,
                ITYPE_t lsize):
  cdef unsigned int count = 0
  cdef DTYPE_t m
  cdef unsigned int c,offset
  cdef DTYPE_t layers = layers1
  cdef unsigned int map_dofs = <unsigned int>map_dofs1
  cdef unsigned int wedges = <unsigned int>wedges1
  cdef unsigned int lins = <unsigned int>lins1
  cdef unsigned int mm,d,i,j,k,l
  cdef np.ndarray[DTYPE_t, ndim=1] ind = np.zeros(lsize, dtype=DTYPE)
  cdef DTYPE_t a1,a2,a3
  cdef int a4
  cdef int len1 = len(mesh2d)
  cdef int len2



  for mm in range(0,lins):
    offset = 0
    for d in range(0,2):
      c = 0
      for i in range(0,len1):
        a4 = dofs[i, d]
        if a4 != 0:
          len2 = len(A[d])
          for j in range(0, mesh2d[i]):
            m = mapp[mm][c]
            for k in range(0, len2):
              a3 = <DTYPE_t>A[d][k]*a4
              for l in range(0,wedges):
                      ind[count + l * nums[2]*map_dofs] = l + m*a4*(layers - d) + a3 + offset
              count+=1
            c+=1
        elif a4 != 0:
          c+= <unsigned int>mesh2d[i]
        offset += a4*nums[i]*(layers - d)
  return ind


@cython.boundscheck(False)
def compute_ind_extr(np.ndarray[DTYPE_t, ndim=1] nums,
                ITYPE_t map_dofs1,
                ITYPE_t lins1,
                DTYPE_t layers1,
                np.ndarray[DTYPE_t, ndim=1] mesh2d,
                np.ndarray[DTYPE_t, ndim=2] dofs not None,
                A not None,
                ITYPE_t wedges1,
                mapp,
                ITYPE_t lsize):
  cdef unsigned int count = 0
  cdef DTYPE_t m
  cdef unsigned int c,offset
  cdef DTYPE_t layers = layers1
  cdef unsigned int map_dofs = <unsigned int>map_dofs1
  cdef unsigned int wedges = <unsigned int>wedges1
  cdef unsigned int lins = <unsigned int>lins1
  cdef unsigned int mm,d,i,j,k,l
  cdef np.ndarray[DTYPE_t, ndim=1] ind = np.zeros(lsize, dtype=DTYPE)
  cdef DTYPE_t a1,a2,a3
  cdef int a4
  cdef int len1 = len(mesh2d)
  cdef int len2



  for mm in range(0,lins):
    offset = 0
    for d in range(0,2):
      c = 0
      for i in range(0,len1):
        a4 = dofs[i, d]
        if a4 != 0:
          len2 = len(A[d])
          for j in range(0, mesh2d[i]):
            m = mapp[mm][c]
            for k in range(0, len2):
              ind[count] = m*a4*(layers - d) + <DTYPE_t>A[d][k]*a4 + offset
              count+=1
            c+=1
        elif a4 != 0:
          c+= <unsigned int>mesh2d[i]
        offset += a4*nums[i]*(layers - d)
  return ind



#cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, int t):
#    """Return an array of SIZE elements (each of type T) with data from PTR."""
#    return np.PyArray_SimpleNewFromData(1, &size, t, ptr)

#cdef ext
#def compute_ind2(nums,map_dofs,lins,layers,mesh2d,dofs,A,wedges,mapp,lsize):
#    cdef int *ind_data
#    compute_ind_function(nums, map_dofs, lins, layers, mesh2d, dofs, &ind_data)
#    return data_to_numpy_array_with_spec(ind_data, lsize, np.NPY_INT32)