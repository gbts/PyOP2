# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Wrap OP2 library for PyOP2

The C level OP2 runtime needs to be aware of the data structures that
the python layer is managing.  So that things like plan construction
and halo swapping actually have some data to deal with.  Equally, the
python level objects need to keep a hold of their C layer counterparts
for interoperability.  All this interfacing is dealt with here.

Naming conventions:

Wrappers around C functions use the same names as in the OP2-Common
library.  Hence, the python classes corresponding to C structs are not
opSet, opDat and so forth, but rather op_set and op_dat.

How it works:

A python object that has a C counterpart has a slot named
_lib_handle.  This is either None, meaning the C initialiser has not
yet been called, or else a handle to the Cython class wrapping the C
data structure.  This handle is exposed to the Cython layer through
the _c_handle property which takes care of instantiating the C layer
object if it does not already exist.

To get this interfacing library, do something like:

    import op_lib_core as core

The C data structure is built on demand when asking for the handle
through the _c_handle property.

C layer function calls that require an OP2 object as an argument are
wrapped such that you don't need to worry about passing the handle,
instead, just pass the python object.  That is, you do:

   core.op_function(set)

not

   core.op_function(set._c_handle)

Most C level objects are completely opaque to the python layer.  The
exception is the op_plan structure, whose data must be marshalled to
the relevant device on the python side.  The slots of the op_plan
struct are exposed as properties to python.  Thus, to get the ind_map
array from a plan you do:

   plan = core.op_plan(kernel, set, *args)

   ind_map = plan.ind_map

Scalars are returned as scalars, arrays are wrapped in a numpy array
of the appropriate size.

WARNING, the arrays returned by these properties have their data
buffer pointing to the C layer's data.  As such, they should be
considered read-only.  If you modify them on the python side, the plan
will likely be wrong.

TODO:
Cleanup of C level datastructures is currently not handled.
"""

from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from cpython cimport bool
import base
import numpy as np
cimport numpy as np
cimport _op_lib_core as core

np.import_array()

cdef data_to_numpy_array_with_template(void * ptr, arr):
    """Return an array with the same properties as ARR with data from PTR."""
    cdef np.npy_intp dim = np.size(arr)
    cdef np.dtype t = arr.dtype
    shape = np.shape(arr)
    return np.PyArray_SimpleNewFromData(1, &dim, t.type_num, ptr).reshape(shape)

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, int t):
    """Return an array of SIZE elements (each of type T) with data from PTR."""
    return np.PyArray_SimpleNewFromData(1, &size, t, ptr)

cdef dlopen_openmpi():
    cdef void * handle = NULL
    cdef int mode = core.RTLD_NOW | core.RTLD_GLOBAL | core.RTLD_NOLOAD
    cdef char * libname
    core.emit_ifdef()
    for name in ['libmpi.so', 'libmpi.so.0', 'libmpi.so.1',
                    'libmpi.dylib', 'libmpi.0.dylib', 'libmpi.1.dylib']:
        libname = name
        handle = core.dlopen(libname, mode)
        if handle is not NULL:
            break
    core.emit_endif()

def op_init(args, diags):
    """Initialise OP2

ARGS should be a list of strings to pass as "command-line" arguments
DIAGS should be an integer specifying the diagnostic level.  The
larger it is, the more chatty OP2 will be."""
    cdef char **argv
    cdef int diag_level = diags
    dlopen_openmpi()
    if args is None:
        core.op_init(0, NULL, diag_level)
        return
    args = [bytes(x) for x in args]
    argv = <char **>malloc(sizeof(char *) * len(args))
    if argv is NULL:
        raise MemoryError()
    try:
        for i, a in enumerate(args):
            argv[i] = a
        core.op_init(len(args), argv, diag_level)
    finally:
        # We can free argv here, because op_init_core doesn't keep a
        # handle to the arguments.
        free(argv)

def op_exit():
    """Clean up C level data"""
    core.op_rt_exit()
    core.op_exit()

cdef class op_set:
    cdef core.op_set _handle
    def __cinit__(self, set):
        """Instantiate a C-level op_set from SET"""
        cdef int size = set.size
        cdef char * name = set.name
        self._handle = core.op_decl_set_core(size, name)
        # The C constructor does not set the other size attributes (this is
        # only done when using libop2_mpi, which we're not using)
        # Note the exclusive semantics for exec_size and nonexec_size used in
        # the core library, which are different from the PyOP2 semantics
        self._handle.core_size = set.core_size
        self._handle.exec_size = set.exec_size - set.size
        self._handle.nonexec_size = set.total_size - set.exec_size

    @property
    def size(self):
        """Return the number of elements in the set"""
        return self._handle.size

    @property
    def core_size(self):
        """Return the number of core elements (MPI-only)"""
        return self._handle.core_size

    @property
    def exec_size(self):
        """Return the number of additional imported elements to be executed"""
        return self._handle.exec_size

    @property
    def nonexec_size(self):
        """Return the number of additional imported elements that are not executed"""
        return self._handle.nonexec_size

cdef class op_dat:
    cdef core.op_dat _handle
    def __cinit__(self, dat):
        """Instantiate a C-level op_dat from DAT"""
        cdef op_set set = dat.dataset._c_handle
        cdef int dim = dat.cdim
        cdef int size = dat.dtype.itemsize
        cdef char * type
        cdef np.ndarray data
        cdef char * dataptr
        cdef char * name = dat.name
        tmp = dat.ctype + ":soa" if dat.soa else ""
        type = tmp
        if len(dat._data) > 0:
            data = dat.data
            dataptr = <char *>np.PyArray_DATA(data)
        else:
            dataptr = <char *>NULL
        self._handle = core.op_decl_dat_core(set._handle, dim, type,
                                             size, dataptr, name)

cdef class op_map:
    cdef core.op_map _handle
    def __cinit__(self, map):
        """Instantiate a C-level op_map from MAP"""
        cdef op_set frm = map.iterset._c_handle
        cdef op_set to = map.dataset._c_handle
        cdef int dim = map.dim
        cdef np.ndarray values = map.values
        cdef char * name = map.name
        if values.size == 0:
            self._handle = core.op_decl_map_core(frm._handle, to._handle,
                                                 dim, NULL, name)
        else:
            self._handle = core.op_decl_map_core(frm._handle, to._handle, dim,
                                                 <int *>np.PyArray_DATA(values), name)

cdef class op_arg:
    cdef core.op_arg _handle
    def __cinit__(self, arg):
        """Instantiate a C-level op_arg from ARG."""
        cdef int idx
        cdef op_map map
        cdef core.op_map _map
        cdef int dim
        cdef int size
        cdef char * type
        cdef core.op_access acc
        cdef np.ndarray data
        cdef op_dat _dat

        # Map Python-layer access descriptors down to C enum
        acc = {'READ'  : core.OP_READ,
               'WRITE' : core.OP_WRITE,
               'RW'    : core.OP_RW,
               'INC'   : core.OP_INC,
               'MIN'   : core.OP_MIN,
               'MAX'   : core.OP_MAX}[arg.access._mode]

        if isinstance(arg.data, base.Dat):
            _dat = arg.data._c_handle
            if arg._is_indirect:
                idx = arg.idx
                map = arg.map._c_handle
                _map = map._handle
            else:
                idx = -1
                _map = <core.op_map>NULL
            dim = arg.data.cdim
            type = arg.ctype
            self._handle = core.op_arg_dat_core(_dat._handle, idx, _map,
                                                dim, type, acc)
        elif isinstance(arg.data, base.Global):
            dim = arg.data.cdim
            size = arg.data.data.size/dim
            type = arg.ctype
            data = arg.data.data
            self._handle = core.op_arg_gbl_core(<char *>np.PyArray_DATA(data), dim,
                                                type, size, acc)

cdef class op_plan:
    cdef int idx
    cdef int set_size
    cdef int nind_ele
    def __cinit__(self, kernel, iset, *args, **kwargs):
        """Instantiate a C-level op_plan for a parallel loop.

Arguments to this constructor should be the arguments of the parallel
loop, i.e. the KERNEL, the ISET (iteration set) and any
further ARGS."""
        cdef op_set _set = iset._c_handle
        cdef char * name = kernel.name
        cdef int part_size = kwargs.get('partition_size', 1)
        cdef int nargs = len(args)
        cdef op_arg _arg
        cdef core.op_arg *_args
        cdef int ninds
        cdef int *inds
        cdef int i
        cdef int ind = 0

        self.set_size = _set.size
        # Size of the plan is incremented by the exec_size if any
        # argument is indirect and not read-only.  exec_size is only
        # ever non-zero in an MPI setting.
        if any(arg._is_indirect_and_not_read for arg in args):
            self.set_size += _set.exec_size

        # Count number of indirect arguments.  This will need changing
        # once we deal with vector maps.
        self.nind_ele = sum(arg._is_indirect for arg in args)

        # Build list of args to pass to C-level op_plan function.
        _args = <core.op_arg *>malloc(nargs * sizeof(core.op_arg))
        if _args is NULL:
            raise MemoryError()
        inds = <int *>malloc(nargs * sizeof(int))
        if inds is NULL:
            raise MemoryError()
        try:
            # _args[i] is the ith argument
            # inds[i] is:
            #   -1 if the ith argument is direct
            #   n >= 0 if the ith argument is indirect
            #    where n counts the number of unique indirect dats.
            #    thus, if there are two arguments, both indirect but
            #    both referencing the same dat/map pair (with
            #    different indices) then ninds = {0,0}
            ninds = 0
            # Keep track of which indirect args we've already seen to
            # get value of inds correct.
            d = {}
            for i in range(nargs):
                inds[i] = -1    # Assume direct
                arg = args[i]
                _arg = arg._c_handle
                _args[i] = _arg._handle
                # Fix up inds[i] in indirect case
                if arg._is_indirect:
                    if d.has_key((arg._dat,arg._map)):
                        inds[i] = d[(arg._dat,arg._map)]
                    else:
                        inds[i] = ind
                        d[(arg._dat,arg._map)] = ind
                        ind += 1
                        ninds += 1
            core.op_plan_core(name, _set._handle,
                              part_size, nargs, _args,
                              ninds, inds)
            self.idx = core.OP_plan_index - 1
        finally:
            # We can free these because op_plan_core doesn't keep a
            # handle to them.
            free(_args)
            free(inds)

    cdef core.op_plan *_handle(self):
        return &core.OP_plans[self.idx]

    @property
    def ninds(self):
        """Return the number of unique indirect arguments"""
        return self._handle().ninds

    @property
    def nargs(self):
        """Return the total number of arguments"""
        return self._handle().nargs

    @property
    def part_size(self):
        """Return the partition size.

Normally this will be zero, indicating that the plan should guess the
best partition size."""
        return self._handle().part_size

    @property
    def nthrcol(self):
        """The number of thread colours in each block.

There are nblocks blocks so nthrcol[i] gives the number of colours in
the ith block."""
        cdef int size = self.nblocks
        return data_to_numpy_array_with_spec(self._handle().nthrcol, size, np.NPY_INT32)

    @property
    def thrcol(self):
        """Thread colours of each element.

The ith entry in this array is the colour of ith element of the
iteration set the plan is defined on."""
        cdef int size = self.set_size
        return data_to_numpy_array_with_spec(self._handle().thrcol, size, np.NPY_INT32)

    @property
    def offset(self):
        """The offset into renumbered mappings for each block.

This tells us where in loc_map (q.v.) this block's renumbered mapping
starts."""
        cdef int size = self.nblocks
        return data_to_numpy_array_with_spec(self._handle().offset, size, np.NPY_INT32)

    @property
    def ind_map(self):
        """Renumbered mappings for each indirect dataset.

The ith indirect dataset's mapping starts at:

    ind_map[(i-1) * set_size]

But we need to fix this up for the block we're currently processing,
so see also ind_offs.
"""
        cdef int size = self.set_size * self.nind_ele
        return data_to_numpy_array_with_spec(self._handle().ind_map, size, np.NPY_INT32)

    @property
    def ind_offs(self):
        """Offsets for each block into ind_map (q.v.).

The ith /unique/ indirect dataset's offset is at:

    ind_offs[(i-1) + blockId * N]

where N is the number of unique indirect datasets."""
        cdef int size = self.nblocks * self.ninds
        return data_to_numpy_array_with_spec(self._handle().ind_offs, size, np.NPY_INT32)

    @property
    def ind_sizes(self):
        """The size of each indirect dataset per block.

The ith /unique/ indirect direct has

    ind_sizes[(i-1) + blockID * N]

elements to be staged in, where N is the number of unique indirect
datasets."""
        cdef int size = self.nblocks * self.ninds
        return data_to_numpy_array_with_spec(self._handle().ind_sizes, size, np.NPY_INT32)

    @property
    def nindirect(self):
        """Total size of each unique indirect dataset"""
        cdef int size = self.ninds
        return data_to_numpy_array_with_spec(self._handle().nindirect, size, np.NPY_INT32)

    @property
    def loc_map(self):
        """Local indirect dataset indices, see also offset

Once the ith unique indirect dataset has been copied into shared
memory (via ind_map), this mapping array tells us where in shared
memory the nth iteration element is:

    arg_i_s + loc_map[(i-1) * set_size + n + offset[blockId]] * dim(arg_i)
"""
        cdef int size = self.set_size * self.nind_ele
        return data_to_numpy_array_with_spec(self._handle().loc_map, size, np.NPY_INT16)

    @property
    def nblocks(self):
        """The number of blocks"""
        return self._handle().nblocks

    @property
    def nelems(self):
        """The number of elements in each block"""
        cdef int size = self.nblocks
        return data_to_numpy_array_with_spec(self._handle().nelems, size, np.NPY_INT32)

    @property
    def ncolors_core(self):
        """Number of core (non-halo colours)

MPI only."""
        return self._handle().ncolors_core

    @property
    def ncolors_owned(self):
        """Number of colours for blocks with only owned elements

MPI only."""
        return self._handle().ncolors_owned

    @property
    def ncolors(self):
        """Number of block colours"""
        return self._handle().ncolors

    @property
    def ncolblk(self):
        """Number of blocks for each colour

This array is allocated to be set_size long, but this is the worst
case scenario (every element interacts with every other).  The number
of "real" elements is ncolors."""
        cdef int size = self.set_size
        return data_to_numpy_array_with_spec(self._handle().ncolblk, size, np.NPY_INT32)

    @property
    def blkmap(self):
        """Mapping from device's block ID to plan's block ID.

There are nblocks entries here, you should index into this with the
device's "block" address plus an offset which is

    sum(ncolblk[i] for i in range(0, current_colour))"""
        cdef int size = self.nblocks
        return data_to_numpy_array_with_spec(self._handle().blkmap, size, np.NPY_INT32)

    @property
    def nsharedCol(self):
        """The amount of shared memory required for each colour"""
        cdef int size = self.ncolors
        return data_to_numpy_array_with_spec(self._handle().nsharedCol, size, np.NPY_INT32)

    @property
    def nshared(self):
        """The total number of bytes of shared memory the plan uses"""
        return self._handle().nshared

    @property
    def transfer(self):
        """Data transfer per kernel call"""
        return self._handle().transfer

    @property
    def transfer2(self):
        """Bytes of cache line per kernel call"""
        return self._handle().transfer2

    @property
    def count(self):
        """Number of times this plan has been used"""
        return self._handle().count

def free_sparsity(object sparsity):
    cdef np.ndarray tmp
    for attr in ['_rowptr', '_colidx', '_d_nnz', '_o_nnz']:
        try:
            tmp = getattr(sparsity, attr)
            free(<void *>np.PyArray_DATA(tmp))
        except:
            pass

def build_sparsity(object sparsity, bool parallel):
    cdef int rmult, cmult
    rmult, cmult = sparsity._dims
    cdef int nrows = sparsity._nrows
    cdef int lsize = nrows*rmult
    cdef op_map rmap, cmap
    cdef int nmaps = len(sparsity._rmaps)
    cdef int *d_nnz, *o_nnz, *rowptr, *colidx
    cdef int d_nz, o_nz

    cdef core.op_map *rmaps = <core.op_map *>malloc(nmaps * sizeof(core.op_map))
    if rmaps is NULL:
        raise MemoryError("Unable to allocate space for rmaps")
    cdef core.op_map *cmaps = <core.op_map *>malloc(nmaps * sizeof(core.op_map))
    if cmaps is NULL:
        raise MemoryError("Unable to allocate space for cmaps")

    try:
        for i in range(nmaps):
            rmap = sparsity._rmaps[i]._c_handle
            cmap = sparsity._cmaps[i]._c_handle
            rmaps[i] = rmap._handle
            cmaps[i] = cmap._handle

        if parallel:
            core.build_sparsity_pattern_mpi(rmult, cmult, nrows, nmaps,
                                            rmaps, cmaps, &d_nnz, &o_nnz,
                                            &d_nz, &o_nz)
            sparsity._d_nnz = data_to_numpy_array_with_spec(d_nnz, lsize,
                                                            np.NPY_INT32)
            sparsity._o_nnz = data_to_numpy_array_with_spec(o_nnz, lsize,
                                                            np.NPY_INT32)
            sparsity._rowptr = []
            sparsity._colidx = []
            sparsity._d_nz = d_nz
            sparsity._o_nz = o_nz
        else:
            core.build_sparsity_pattern_seq(rmult, cmult, nrows, nmaps,
                                            rmaps, cmaps,
                                            &d_nnz, &rowptr, &colidx, &d_nz)
            sparsity._d_nnz = data_to_numpy_array_with_spec(d_nnz, lsize,
                                                            np.NPY_INT32)
            sparsity._o_nnz = []
            sparsity._rowptr = data_to_numpy_array_with_spec(rowptr, lsize+1,
                                                            np.NPY_INT32)
            sparsity._colidx = data_to_numpy_array_with_spec(colidx,
                                                            rowptr[lsize],
                                                            np.NPY_INT32)
            sparsity._d_nz = d_nz
            sparsity._o_nz = 0
    finally:
        free(rmaps)
        free(cmaps)

include "plan.pyx"
