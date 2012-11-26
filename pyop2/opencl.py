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

"""OP2 OpenCL backend."""

from device import *
import device as op2
from utils import verify_reshape, uniquify, maybe_setflags
import configuration as cfg
import pyopencl as cl
from pyopencl import array
import pkg_resources
import pycparser
import numpy as np
import collections
import warnings
import math
from jinja2 import Environment, PackageLoader
from pycparser import c_parser, c_ast, c_generator
import os
import re
import time
import md5

class Kernel(op2.Kernel):
    """OP2 OpenCL kernel type."""

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

    class Instrument(c_ast.NodeVisitor):
        """C AST visitor for instrumenting user kernels.
             - adds memory space attribute to user kernel declaration
             - appends constant declaration to user kernel param list
             - adds a separate function declaration for user kernel
        """
        def instrument(self, ast, kernel_name, instrument, constants):
            self._kernel_name = kernel_name
            self._instrument = instrument
            self._ast = ast
            self._constants = constants
            self.generic_visit(ast)
            idx = ast.ext.index(self._func_node)
            ast.ext.insert(0, self._func_node.decl)

        def visit_FuncDef(self, node):
            if node.decl.name == self._kernel_name:
                self._func_node = node
                self.visit(node.decl)

        def visit_ParamList(self, node):
	    print self._instrument
	    print node.params
	    #print "PARAM: params = %d " % len(node.params)
            for i, p in enumerate(node.params):
	        #print self._instrument[i][0]
                if self._instrument[i][0]:
                    p.storage.append(self._instrument[i][0])
                if self._instrument[i][1]:
                    p.type.quals.append(self._instrument[i][1])

            #print "PARAM: constants = %d " % len(self._constants)
            for cst in self._constants:
                if cst._is_scalar:
                    t = c_ast.TypeDecl(cst._name, [], c_ast.IdentifierType([cst._cl_type]))
                else:
                    t = c_ast.PtrDecl([], c_ast.TypeDecl(cst._name, ["__constant"], c_ast.IdentifierType([cst._cl_type])))
                decl = c_ast.Decl(cst._name, [], [], [], t, None, 0)
                node.params.append(decl)

    def instrument(self, instrument, constants):
        def comment_remover(text):
            """Remove all C- and C++-style comments from a string."""
            # Reference: http://stackoverflow.com/questions/241327/python-snippet-to-remove-c-and-c-comments
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return ""
                else:
                    return s
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                                 re.DOTALL | re.MULTILINE)
            return re.sub(pattern, replacer, text)

        ast = c_parser.CParser().parse(comment_remover(self._code).replace("\\\n", "\n"))
        Kernel.Instrument().instrument(ast, self._name, instrument, constants)
        return c_generator.CGenerator().visit(ast)

class Arg(op2.Arg):
    """OP2 OpenCL argument type."""

    # FIXME actually use this in the template
    def _indirect_kernel_arg_name(self, idx):
        if self._is_global:
            if self._is_global_reduction:
                return self._reduction_local_name
            else:
                return self._name
        if self._is_direct:
            if self.data.soa:
                return "%s + (%s + offset_b)" % (self._name, idx)
            return "%s + (%s + offset_b) * %s" % (self._name, idx,
                                                  self.data.cdim)
        if self._is_indirect:
            if self._is_vec_map:
                return self._vec_name
            if self.access is op2.INC:
                return self._local_name()
            else:
                return "%s + loc_map[%s * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect, idx,
                       self.data.cdim)

    def _direct_kernel_arg_name(self, idx=None):
        if self._is_mat:
            return self._mat_entry_name
        if self._is_staged_direct:
            return self._local_name()
        elif self._is_global_reduction:
            return self._reduction_local_name
        elif self._is_global:
            return self._name
        else:
            return "%s + %s" % (self._name, idx)

class DeviceDataMixin(op2.DeviceDataMixin):
    """Codegen mixin for datatype and literal translation."""

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero', 'min', 'max'])
    CL_TYPES = {np.dtype('uint8'): ClTypeInfo('uchar', '0', '0', '255'),
                np.dtype('int8'): ClTypeInfo('char', '0', '-127', '127'),
                np.dtype('uint16'): ClTypeInfo('ushort', '0', '0', '65535'),
                np.dtype('int16'): ClTypeInfo('short', '0', '-32767', '32767'),
                np.dtype('uint32'): ClTypeInfo('uint', '0u', '0u', '4294967295u'),
                np.dtype('int32'): ClTypeInfo('int', '0', '-2147483647', '2147483647'),
                np.dtype('uint64'): ClTypeInfo('ulong', '0ul', '0ul', '18446744073709551615ul'),
                np.dtype('int64'): ClTypeInfo('long', '0l', '-9223372036854775807l', '9223372036854775807l'),
                np.dtype('float32'): ClTypeInfo('float', '0.0f', '-3.4028235e+38f', '3.4028235e+38f'),
                np.dtype('float64'): ClTypeInfo('double', '0.0', '-1.7976931348623157e+308', '1.7976931348623157e+308')}

    def _allocate_device(self):
        if self.state is DeviceDataMixin.DEVICE_UNALLOCATED:
            if self.soa:
                shape = self._data.T.shape
            else:
                shape = self._data.shape
            self._device_data = array.empty(_queue, shape=shape,
                                            dtype=self.dtype)
            self.state = DeviceDataMixin.HOST

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.HOST:
            self._device_data.set(self._maybe_to_soa(self._data),
                                  queue=_queue)
        self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        flag = self._data.flags['WRITEABLE']
        maybe_setflags(self._data, write=True)
        if self.state is DeviceDataMixin.DEVICE:
            self._device_data.get(_queue, self._data)
            self._data = self._maybe_to_aos(self._data)
            self.state = DeviceDataMixin.BOTH
        maybe_setflags(self._data, write=flag)

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].zero

    @property
    def _cl_type_min(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].min

    @property
    def _cl_type_max(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].max

class Dat(op2.Dat, DeviceDataMixin):
    """OP2 OpenCL vector data type."""

    _arg_type = Arg


    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.sqrt(array.dot(self.array, self.array).get())

def solve(M, b, x):
    x._from_device()
    b._from_device()
    core.solve(M, b, x)
    x._to_device()

class Mat(op2.Mat, DeviceDataMixin):
    """OP2 OpenCL matrix data type."""

    _arg_type = Arg

    def _allocate_device(self):
        pass

    def _to_device(self):
        pass

    def _from_device(self):
        pass

    @property
    def _dev_array(self):
        if not hasattr(self, '__dev_array'):
            setattr(self, '__dev_array',
                    array.empty(_queue,
                                self._sparsity._c_handle.total_nz,
                                self.dtype))
        return getattr(self, '__dev_array')

    @property
    def _dev_colidx(self):
        if not hasattr(self, '__dev_colidx'):
            setattr(self, '__dev_colidx',
                    array.to_device(_queue,
                                    self._sparsity._c_handle.colidx))
        return getattr(self, '__dev_colidx')

    @property
    def _dev_rowptr(self):
        if not hasattr(self, '__dev_rowptr'):
            setattr(self, '__dev_rowptr',
                    array.to_device(_queue,
                                    self._sparsity._c_handle.rowptr))
        return getattr(self, '__dev_rowptr')

    def _upload_array(self):
        self._dev_array.set(self._c_handle.array, queue=_queue)
        self.state = DeviceDataMixin.BOTH

    def assemble(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._dev_array.get(queue=_queue, ary=self._c_handle.array)
            self._c_handle.restore_array()
            self.state = DeviceDataMixin.BOTH
        self._c_handle.assemble()

    @property
    def cdim(self):
        return np.prod(self.dims)

class Const(op2.Const, DeviceDataMixin):
    """OP2 OpenCL data that is constant for any element of any set."""

    @property
    def _array(self):
        if not hasattr(self, '__array'):
            setattr(self, '__array', array.to_device(_queue, self._data))
        return getattr(self, '__array')

class Global(op2.Global, DeviceDataMixin):
    """OP2 OpenCL global value."""

    _arg_type = Arg

    @property
    def _array(self):
        if not hasattr(self, '_device_data'):
            self._device_data = array.to_device(_queue, self._data)
        return self._device_data

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros (nelems * self.cdim, dtype=self.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        cl.enqueue_copy(_queue, self._d_reduc_buffer, self._h_reduc_array, is_blocking=True).wait()

    @property
    def data(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._array.get(_queue, ary=self._data)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

    def _post_kernel_reduction_task(self, nelems, reduction_operator):
        assert reduction_operator in [INC, MIN, MAX]

        def generate_code():
            def headers():
                if self.dtype == np.dtype('float64'):
                    return """
#if defined(cl_khr_fp64)
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

"""
                else:
                    return ""

            def op():
                if reduction_operator is INC:
                    return "INC"
                elif reduction_operator is MIN:
                    return "min"
                elif reduction_operator is MAX:
                        return "max"
                assert False

            return """
%(headers)s
#define INC(a,b) ((a)+(b))
__kernel
void global_%(type)s_%(dim)s_post_reduction (
  __global %(type)s* dat,
  __global %(type)s* tmp,
  __private int count
)
{
  __private %(type)s accumulator[%(dim)d];
  for (int j = 0; j < %(dim)d; ++j)
  {
    accumulator[j] = dat[j];
  }
  for (int i = 0; i < count; ++i)
  {
    for (int j = 0; j < %(dim)d; ++j)
    {
      accumulator[j] = %(op)s(accumulator[j], *(tmp + i * %(dim)d + j));
    }
  }
  for (int j = 0; j < %(dim)d; ++j)
  {
    dat[j] = accumulator[j];
  }
}
""" % {'headers': headers(), 'dim': self.cdim, 'type': self._cl_type, 'op': op()}


        if not _reduction_task_cache.has_key((self.dtype, self.cdim, reduction_operator)):
            _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)] = generate_code()

        src = _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)]
        name = "global_%s_%s_post_reduction" % (self._cl_type, self.cdim)
        prg = cl.Program(_ctx, src).build(options="-Werror")
        kernel = prg.__getattr__(name)
        kernel.append_arg(self._array.data)
        kernel.append_arg(self._d_reduc_buffer)
        kernel.append_arg(np.int32(nelems))
        cl.enqueue_task(_queue, kernel).wait()

        del self._d_reduc_buffer

class Map(op2.Map):
    """OP2 OpenCL map, a relation between two Sets."""

    _arg_type = Arg

    def _to_device(self):
        if not hasattr(self, '_device_values'):
            self._device_values = array.to_device(_queue, self._values)
        else:
            from warnings import warn
            warn("Copying Map data for %s again, do you really want to do this?" % \
                 self)
            self._device_values.set(_queue, self._values)

    def _off_to_device(self):
        if not hasattr(self, '_off_device_values'):
            self._off_device_values = array.to_device(_queue, self.off)
        else:
            from warnings import warn
            warn("Copying Map OFFSET data for %s again, do you really want to do this?" % \
                 self)
            self._off_device_values.set(self.off, queue=_queue)

class Plan(op2.Plan):
    @property
    def ind_map(self):
        if not hasattr(self, '_ind_map'):
            self._ind_map = array.to_device(_queue, super(Plan, self).ind_map)
        return self._ind_map

    @property
    def loc_map(self):
        if not hasattr(self, '_loc_map'):
            self._loc_map = array.to_device(_queue, super(Plan, self).loc_map)
        return self._loc_map

    @property
    def ind_sizes(self):
        if not hasattr(self, '_ind_sizes'):
            self._ind_sizes = array.to_device(_queue, super(Plan, self).ind_sizes)
        return self._ind_sizes

    @property
    def ind_offs(self):
        if not hasattr(self, '_ind_offs'):
            self._ind_offs = array.to_device(_queue, super(Plan, self).ind_offs)
        return self._ind_offs

    @property
    def blkmap(self):
        if not hasattr(self, '_blkmap'):
            self._blkmap = array.to_device(_queue, super(Plan, self).blkmap)
        return self._blkmap

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            self._offset = array.to_device(_queue, super(Plan, self).offset)
        return self._offset

    @property
    def nelems(self):
        if not hasattr(self, '_nelems'):
            self._nelems = array.to_device(_queue, super(Plan, self).nelems)
        return self._nelems

    @property
    def nthrcol(self):
        if not hasattr(self, '_nthrcol'):
            self._nthrcol = array.to_device(_queue, super(Plan, self).nthrcol)
        return self._nthrcol

    @property
    def thrcol(self):
        if not hasattr(self, '_thrcol'):
            self._thrcol = array.to_device(_queue, super(Plan, self).thrcol)
        return self._thrcol

class ParLoop(op2.ParLoop):
    @property
    def _matrix_args(self):
        return [a for a in self.args if a._is_mat]

    @property
    def _unique_matrix(self):
        return uniquify(a.data for a in self._matrix_args)

    @property
    def _matrix_entry_maps(self):
        """Set of all mappings used in matrix arguments."""
        return uniquify(m for arg in self.args  if arg._is_mat for m in arg.map)

    def dump_gen_code(self):
        if cfg['dump-gencode']:
            path = cfg['dump-gencode-path'] % {"kernel": self.kernel.name,
                                               "time": time.strftime('%Y-%m-%d@%H:%M:%S')}

            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(self._src)

    def _i_partition_size(self,layers=None):
        #TODO FIX: something weird here
        #available_local_memory
        warnings.warn('temporary fix to available local memory computation (-512)')
        available_local_memory = _max_local_memory - 512
        # 16bytes local mem used for global / local indices and sizes
        available_local_memory -= 16
        # (4/8)ptr size per dat passed as argument (dat)
        available_local_memory -= (_address_bits / 8) * (len(self._unique_dat_args) + len(self._all_global_non_reduction_args))
        # (4/8)ptr size per dat/map pair passed as argument (ind_map)
        available_local_memory -= (_address_bits / 8) * len(self._unique_indirect_dat_args)
        # (4/8)ptr size per global reduction temp array
        available_local_memory -= (_address_bits / 8) * len(self._all_global_reduction_args)
        # (4/8)ptr size per indirect arg (loc_map)
        available_local_memory -= (_address_bits / 8) * len(self._all_indirect_args)
        # (4/8)ptr size * 7: for plan objects
        available_local_memory -= (_address_bits / 8) * 7
        # 1 uint value for block offset
        available_local_memory -= 4
        # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
        available_local_memory -= 7
        # 12: shared_memory_offset, active_thread_count, active_thread_count_ceiling variables (could be 8 or 12 depending)
        #     and 3 for potential padding after shared mem buffer
        available_local_memory -= 12 + 3
        # 2 * (4/8)ptr size + 1uint32: DAT_via_MAP_indirection(./_size/_map) per dat map pairs
        available_local_memory -= 4 + (_address_bits / 8) * 2 * len(self._unique_indirect_dat_args)
        # inside shared memory padding
        available_local_memory -= 2 * (len(self._unique_indirect_dat_args) - 1)

	#k = 0
	#print len(self._all_indirect_args)
	#for a in self._all_indirect_args:
	#  k+=1
	#  #print a
	#  print k, a.data._bytes_per_elem

	#extruded case:
	#	1. The MAPS are flattened so for example for a map 1->10, 10 "args" are considered
	#	2. The max_bytes below represents the number of bytes for each iteration element of the parloop
	#	   for example for a WEDGE when we loop over triangles
	#	3. TODO: divide by the size of a column, return the number of columns in each partition
        max_bytes = sum(map(lambda a: a.data._bytes_per_elem, self._all_indirect_args))
        if layers > 1:
	  max_bytes *= layers
	  print "max bytes for 3D case = %d " % max_bytes
        #returns the number of elements in a partition
        return available_local_memory / (2 * _warpsize * max_bytes) * (2 * _warpsize)

    def launch_configuration(self, layers=None):
        if self._is_direct:
            per_elem_max_local_mem_req = self._max_shared_memory_needed_per_set_element
            shared_memory_offset = per_elem_max_local_mem_req * _warpsize
            if per_elem_max_local_mem_req == 0:
                wgs = _max_work_group_size
            else:
                # 16bytes local mem used for global / local indices and sizes
                # (4/8)ptr bytes for each dat buffer passed to the kernel
                # (4/8)ptr bytes for each temporary global reduction buffer passed to the kernel
                # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
                warnings.warn('temporary fix to available local memory computation (-512)')
                available_local_memory = _max_local_memory - 512
                available_local_memory -= 16
                available_local_memory -= (len(self._unique_dat_args) + len(self._all_global_non_reduction_args))\
                                          * (_address_bits / 8)
                available_local_memory -= len(self._all_global_reduction_args) * (_address_bits / 8)
                available_local_memory -= 7
                ps = available_local_memory / per_elem_max_local_mem_req
                wgs = min(_max_work_group_size, (ps / _warpsize) * _warpsize)
            nwg = min(_pref_work_group_count, int(math.ceil(self._it_space.size / float(wgs))))
            ttc = wgs * nwg

            local_memory_req = per_elem_max_local_mem_req * wgs
            return {'thread_count': ttc,
                    'work_group_size': wgs,
                    'work_group_count': nwg,
                    'local_memory_size': local_memory_req,
                    'local_memory_offset': shared_memory_offset}
        else:
            return {'partition_size': self._i_partition_size(layers)}

    def codegen(self, conf):
        def instrument_user_kernel():
            inst = []
	    #print len(self.args)
            for arg in self.args:
                i = None
                if self._is_direct:
                    if (arg._is_direct and (arg.data._is_scalar or arg.data.soa)) or\
                       (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    else:
                        i = ("__private", None)
                else: # indirect loop
                    if arg._is_direct or (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    elif (arg._is_indirect or arg._is_vec_map) and not arg._is_indirect_reduction:
                        i = ("__local", None)
                    else:
                        i = ("__private", None)

                inst.append(i)

            for i in self._it_space.extents:
                inst.append(("__private", None))

            if self._it_space.layers > 1:
		inst.append(("__private", None))

            return self._kernel.instrument(inst, Const._definitions())

        # check cache
        key = self._cache_key
        self._src = op2._parloop_cache.get(key)
        if self._src is not None:
            return

        #do codegen
        user_kernel = instrument_user_kernel()
        template = _jinja2_direct_loop if self._is_direct \
                                       else _jinja2_indirect_loop

        self._src = template.render({'parloop': self,
                                     'user_kernel': user_kernel,
                                     'launch': conf,
                                     'codegen': {'amd': _AMD_fixes},
                                     'op2const': Const._definitions()
                                 }).encode("ascii")
        self.dump_gen_code()
        print "##################################################################################### START"
        print self._src
        print "##################################################################################### END"


        ##FOR GPU
        self._src = """
/* Launch configuration:
 *   work group size     : 1
 *   partition size      : 1
 *   local memory size   : 256
 *   local memory offset :
 *   warpsize            : 1
 */

#if defined(cl_khr_fp64)
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define OP_WARPSIZE 1
#define OP2_STRIDE(arr, idx) ((arr)[op2stride * (idx)])

__kernel
void g_reduction_kernel (
  __global double *reduction_result,
  __private double input_value,
  __local double *reduction_tmp_array
) {
  barrier(CLK_LOCAL_MEM_FENCE);
  int lid = get_local_id(0);
  reduction_tmp_array[lid] = input_value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = 1; offset < (int)get_local_size(0); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if(((lid & mask) == 0) && (lid + offset < (int)get_local_size(0))) {
      reduction_tmp_array[lid] += reduction_tmp_array[lid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    *reduction_result = reduction_tmp_array[0];
}


void comp_vol(__private double A[1], __local double *x[], __local double *y[], __private int j);
void comp_vol(__private double A[1], __local double *x[], __local double *y[], __private int j)
{
  for (int i = 0; i < 6; i++)
  {
    A[0] += x[i][0] + y[i][0];
  }

  for (int i = 0; i < 0; i++)
  {
    A[0] += x[i][1] + y[i][1];
  }

}


__kernel
__attribute__((reqd_work_group_size(10, 1, 1)))
void __comp_vol_stub(
  __global double* coords,
  __global double* speed,
  __global double* g,
  int set_size,
  __global int* p_ind_map,
  __global short *p_loc_map,
  __global int* p_ind_sizes,
  __global int* p_ind_offsets,
  __global int* p_blk_map,
  __global int* p_offset,
  __global int* p_nelems,
  __global int* p_nthrcol,
  __global int* p_thrcol,
  __private int block_offset
) {
  __local char shared [2560] __attribute__((aligned(sizeof(long))));
  __local int shared_memory_offset;
  __local int active_threads_count;

  int nbytes;
  int block_id;

  int i_1;
  int j;
  int k;

  // reduction args
  // global reduction local declarations

  double g_reduction_local[1];

  // shared indirection mappings
  __global int* __local coords_map;
  __local int coords_size;
  __local double* __local coords_shared;
  __global int* __local speed_map;
  __local int speed_size;
  __local double* __local speed_shared;

  __local double* coords_vec[6];
  __local double* speed_vec[6];
  __local double* coords_vec2[6];
  __local double* speed_vec2[6];
  __local int off[15] = {1,1,1 ,1,1,1, 1,1,1, 1,1,1, 1,1,1};

  //printf(" --------------------------------------------------------------- START %d \\n",get_local_id(0));

  if (get_local_id(0) == 0) {
    block_id = p_blk_map[get_group_id(0) + block_offset];
    active_threads_count = p_nelems[block_id];
    shared_memory_offset = p_offset[block_id];
    //printf("shared_memory_offset = %d \\n",shared_memory_offset);
    coords_size = p_ind_sizes[0 + block_id * 2];
    //printf(" coords_size = %d \\n", coords_size);
    coords_map = &p_ind_map[0 * set_size] + p_ind_offsets[0 + block_id * 2];
    speed_size = p_ind_sizes[1 + block_id * 2];
    //printf(" speed_size = %d \\n", speed_size);
    speed_map = &p_ind_map[6 * set_size] + p_ind_offsets[1 + block_id * 2];

    //printf(" coords_map = %d %d %d %d %d %d\\n", coords_map[0],coords_map[1],coords_map[2],coords_map[3],coords_map[4],coords_map[5]);
    //printf(" speed_map = %d %d %d %d %d %d\\n", speed_map[0],speed_map[1],speed_map[2],speed_map[3],speed_map[4],speed_map[5]);
    //printf("local size = %d \\n", get_local_size(0));

    nbytes = 0;
    coords_shared = (__local double*) (&shared[nbytes]);
    nbytes += ROUND_UP(coords_size * 1 * sizeof(double) * 10);
    speed_shared = (__local double*) (&shared[nbytes]);
    nbytes += ROUND_UP(speed_size * 1 * sizeof(double) * 10);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  //printf(" Step 1 <---> %d \\n",get_local_id(0));

// staging in of indirect dats

  // DO BETTER HERE BY NOT COPYING THE +1 DOFS
  // AS THEY ARE ALREADY PART OF THE COLUMN THAT IS BEING COPIED.
  // THIS APPLIES TO THE DOFS THAT ARE MIRRORRED DOFS NOT TO THE DOFS THAT ARE INTER LAYERS

  for (i_1 = get_local_id(0); i_1 < coords_size * 1; i_1 += get_local_size(0)) {
   for (j = 0; j < 10; j++){
    coords_shared[i_1*10 + j] = coords[coords_map[i_1] + j];
   }
  }

  for (i_1 = get_local_id(0); i_1 < speed_size * 1; i_1 += get_local_size(0)) {
   for (j = 0; j < 10; j++){
    speed_shared[i_1*10 + j] = speed[speed_map[i_1] + j];
   }
  }

  //printf(" Step 2 <---> %d \\n",get_local_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);

  // zeroing private memory for global reduction

  for (i_1 = 0; i_1 < 1; ++i_1) {
    g_reduction_local[i_1] = 0.0;
}


  for (i_1 = get_local_id(0); i_1 < active_threads_count; i_1 += get_local_size(0)) { // 1 = number of threads
      //printf(" set_size = %d \\n", shared_memory_offset);
      //printf(" loc_map = %d %d %d %d %d %d\\n", p_loc_map[0],p_loc_map[1],p_loc_map[2],p_loc_map[3],p_loc_map[4],p_loc_map[5]);
      //printf(" loc_map = %d ", p_loc_map[i_1 + 0*set_size + shared_memory_offset]);
      //printf(" %d ", p_loc_map[i_1 + 1*set_size + shared_memory_offset]);
      //printf(" %d ", p_loc_map[i_1 + 2*set_size + shared_memory_offset]);
      //printf(" %d ", p_loc_map[i_1 + 3*set_size + shared_memory_offset]);
      //printf(" %d ", p_loc_map[i_1 + 4*set_size + shared_memory_offset]);
      //printf(" %d \\n", p_loc_map[i_1 + 5*set_size + shared_memory_offset]);

        // populate vec map
      coords_vec[0] = &coords_shared[p_loc_map[i_1 + 0*set_size + shared_memory_offset] * 10];
      coords_vec[1] = &coords_shared[p_loc_map[i_1 + 1*set_size + shared_memory_offset] * 10];
      coords_vec[2] = &coords_shared[p_loc_map[i_1 + 2*set_size + shared_memory_offset] * 10];
      coords_vec[3] = &coords_shared[p_loc_map[i_1 + 3*set_size + shared_memory_offset] * 10];
      coords_vec[4] = &coords_shared[p_loc_map[i_1 + 4*set_size + shared_memory_offset] * 10];
      coords_vec[5] = &coords_shared[p_loc_map[i_1 + 5*set_size + shared_memory_offset] * 10];

        // populate vec map
      speed_vec[0] = &speed_shared[p_loc_map[i_1 + 6*set_size + shared_memory_offset] * 10];
      speed_vec[1] = &speed_shared[p_loc_map[i_1 + 7*set_size + shared_memory_offset] * 10];
      speed_vec[2] = &speed_shared[p_loc_map[i_1 + 8*set_size + shared_memory_offset] * 10];
      speed_vec[3] = &speed_shared[p_loc_map[i_1 + 9*set_size + shared_memory_offset] * 10];
      speed_vec[4] = &speed_shared[p_loc_map[i_1 + 10*set_size + shared_memory_offset] * 10];
      speed_vec[5] = &speed_shared[p_loc_map[i_1 + 11*set_size + shared_memory_offset] * 10];
    }

  barrier(CLK_LOCAL_MEM_FENCE);

    for(k = 0; k<2; k++){
      for (i_1 = get_local_id(0); i_1 < 10 && i_1 % 2 == k; i_1 += get_local_size(0)) {
        //lid = i_1 % 10;
        //gid = i_1 / 10;

        for(j = 0; j<6; j++){
	  coords_vec2[j] = coords_vec[j] + i_1*off[j];
        }
        for(j = 0; j<6; j++){
	  speed_vec2[j] = speed_vec[j] + i_1*off[j];
        }

	comp_vol(
	  g_reduction_local,
	  coords_vec2,
	  speed_vec2,
	  i_1
        );
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  for (i_1 = 0; i_1 < 1; ++i_1){
      g_reduction_kernel(&g[i_1 + get_group_id(0) * 1], g_reduction_local[i_1], (__local double*) shared);
  }

//printf(" ----------------------------------------------------------------- STOP %d \\n",get_local_id(0));
}
        """







       ## FOR CPU
        self._src = """
/* Launch configuration:
 *   work group size     : 1
 *   partition size      : 1
 *   local memory size   : 256
 *   local memory offset :
 *   warpsize            : 1
 */

#if defined(cl_khr_fp64)
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define OP_WARPSIZE 1
#define OP2_STRIDE(arr, idx) ((arr)[op2stride * (idx)])

__kernel
void g_reduction_kernel (
  __global double *reduction_result,
  __private double input_value,
  __local double *reduction_tmp_array
) {
  barrier(CLK_LOCAL_MEM_FENCE);
  int lid = get_local_id(0);
  reduction_tmp_array[lid] = input_value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = 1; offset < (int)get_local_size(0); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if(((lid & mask) == 0) && (lid + offset < (int)get_local_size(0))) {
      reduction_tmp_array[lid] += reduction_tmp_array[lid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    *reduction_result = reduction_tmp_array[0];
}


void comp_vol(__private double A[1], __local double *x[], __local double *y[], __private int j);
void comp_vol(__private double A[1], __local double *x[], __local double *y[], __private int j){
  for (int i = 0; i < 6; i++){
    A[0] += x[i][0] + y[i][0];
  }

  for (int i = 0; i < 0; i++)
  {
    A[0] += x[i][1] + y[i][1];
  }

}


__kernel
__attribute__((reqd_work_group_size(30, 1, 1)))
void __comp_vol_stub(
  __global double* coords,
  __global double* speed,
  __global double* g,
  int set_size,
  __global int* p_ind_map,
  __global short *p_loc_map,
  __global int* p_ind_sizes,
  __global int* p_ind_offsets,
  __global int* p_blk_map,
  __global int* p_offset,
  __global int* p_nelems,
  __global int* p_nthrcol,
  __global int* p_thrcol,
  __global int* coords_off,
  __global int* speed_off,
  __private int block_offset
) {
  __local char shared [2000] __attribute__((aligned(sizeof(long))));
  __local int shared_memory_offset;
  __local int active_threads_count;

  int nbytes;
  int block_id;

  int i_1;
  int j;
  int k;

  // reduction args
  // global reduction local declarations

  double g_reduction_local[1];

  // shared indirection mappings
  __global int* __local coords_map;
  __local int coords_size;
  __local double* __local coords_shared;
  __global int* __local speed_map;
  __local int speed_size;
  __local double* __local speed_shared;

  __local double* coords_vec[6];
  __local double* speed_vec[6];
  __local double* coords_vec2[6];
  __local double* speed_vec2[6];
  __local int off[15] = {1,1,1 ,1,1,1, 1,1,1, 1,1,1, 1,1,1};

  //printf(" --------------------------------------------------------------- START %d \\n",get_local_id(0));

  if (get_local_id(0) == 0) {
    block_id = p_blk_map[get_group_id(0) + block_offset];
    active_threads_count = p_nelems[block_id];
    shared_memory_offset = p_offset[block_id];
    coords_size = p_ind_sizes[0 + block_id * 2];
    coords_map = &p_ind_map[0 * set_size] + p_ind_offsets[0 + block_id * 2];
    speed_size = p_ind_sizes[1 + block_id * 2];
    speed_map = &p_ind_map[6 * set_size] + p_ind_offsets[1 + block_id * 2];

    nbytes = 0;
    coords_shared = (__local double*) (&shared[nbytes]);
    nbytes += ROUND_UP(coords_size * 1 * sizeof(double) * 10);
    speed_shared = (__local double*) (&shared[nbytes]);
    nbytes += ROUND_UP(speed_size * 1 * sizeof(double) * 10);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

// staging in of indirect dats

  // DO BETTER HERE BY NOT COPYING THE +1 DOFS
  // AS THEY ARE ALREADY PART OF THE COLUMN THAT IS BEING COPIED.
  // THIS APPLIES TO THE DOFS THAT ARE MIRRORRED DOFS NOT TO THE DOFS THAT ARE INTER LAYERS


  for (i_1 = get_local_id(0); i_1 < coords_size * 1; i_1 += get_local_size(0)) {
   for (j = 0; j < 10; j++){ //10 * dim
    coords_shared[i_1*10 + j] = coords[coords_map[i_1] + j];
   }
  }

  for (i_1 = get_local_id(0); i_1 < speed_size * 1; i_1 += get_local_size(0)) {
   for (j = 0; j < 10; j++){
    speed_shared[i_1*10 + j] = speed[speed_map[i_1] + j];
   }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // zeroing private memory for global reduction

  for (i_1 = 0; i_1 < 1; ++i_1) {
    g_reduction_local[i_1] = 0.0;
}


  for (i_1 = get_local_id(0); i_1 < active_threads_count; i_1 += get_local_size(0)) {
        // populate vec map
      coords_vec[0] = &coords_shared[p_loc_map[i_1 + 0*set_size + shared_memory_offset] * 10]; // *10 * dim
      coords_vec[1] = &coords_shared[p_loc_map[i_1 + 1*set_size + shared_memory_offset] * 10];
      coords_vec[2] = &coords_shared[p_loc_map[i_1 + 2*set_size + shared_memory_offset] * 10];
      coords_vec[3] = &coords_shared[p_loc_map[i_1 + 3*set_size + shared_memory_offset] * 10];
      coords_vec[4] = &coords_shared[p_loc_map[i_1 + 4*set_size + shared_memory_offset] * 10];
      coords_vec[5] = &coords_shared[p_loc_map[i_1 + 5*set_size + shared_memory_offset] * 10];

        // populate vec map
      speed_vec[0] = &speed_shared[p_loc_map[i_1 + 6*set_size + shared_memory_offset] * 10];
      speed_vec[1] = &speed_shared[p_loc_map[i_1 + 7*set_size + shared_memory_offset] * 10];
      speed_vec[2] = &speed_shared[p_loc_map[i_1 + 8*set_size + shared_memory_offset] * 10];
      speed_vec[3] = &speed_shared[p_loc_map[i_1 + 9*set_size + shared_memory_offset] * 10];
      speed_vec[4] = &speed_shared[p_loc_map[i_1 + 10*set_size + shared_memory_offset] * 10];
      speed_vec[5] = &speed_shared[p_loc_map[i_1 + 11*set_size + shared_memory_offset] * 10];


      for (i_1 =0; i_1 < 10; i_1 += 1) {

        for(j = 0; j<6; j++){
	  coords_vec[j] += i_1*off[j];
        }
        for(j = 0; j<6; j++){
	  speed_vec[j] += i_1*off[j];
        }

	comp_vol(
	  g_reduction_local,
	  coords_vec,
	  speed_vec,
	  i_1
        );
      }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i_1 = 0; i_1 < 1; ++i_1){
      g_reduction_kernel(&g[i_1 + get_group_id(0) * 1], g_reduction_local[i_1], (__local double*) shared);
  }
}
        """
        op2._parloop_cache[key] = self._src

    def compute(self):
        print "--> 1 <--"
        if self._has_soa:
            op2stride = Const(1, self._it_space.size, name='op2stride',
                              dtype='int32')
        def compile_kernel():
            prg = cl.Program(_ctx, self._src).build(options="-Werror")
            return prg.__getattr__(self._stub_name)

        conf = self.launch_configuration(self._it_space.layers)
        #print "part size = %d" % conf['partition_size']
        print "--> 2 <--"
        if self._is_indirect:
	    if self._it_space.layers > 1:
		#extruded case
		#conf['partition_size'] = conf['partition_size'] #/ (self._it_space.layers - 1)
		#print "number of cols per partition = %d" % conf['partition_size']
		#conf['partition_size'] = 1 # for test purposes - so just 1 column

		self._plan = Plan(self.kernel, self._it_space.iterset,
                              *self._unwound_args,
                              partition_size=conf['partition_size'])

		conf['local_memory_size'] = self._plan.nshared
		conf['ninds'] = self._plan.ninds
		conf['work_group_size'] = min(_max_work_group_size,conf['partition_size'])
		conf['work_group_count'] = self._plan.nblocks
	    else:
		self._plan = Plan(self.kernel, self._it_space.iterset,
                              *self._unwound_args,
                              partition_size=conf['partition_size'])
		conf['local_memory_size'] = self._plan.nshared
		conf['ninds'] = self._plan.ninds
		conf['work_group_size'] = min(_max_work_group_size,conf['partition_size'])
		conf['work_group_count'] = self._plan.nblocks
        conf['warpsize'] = _warpsize
        print "--> 3 <--"
        #print self._it_space.layers
        self.codegen(conf)
        print "--> 4 <--"
        kernel = compile_kernel()
	print "--> 5 <--"
        for arg in self._unique_args:
            arg.data._allocate_device()
            if arg.access is not op2.WRITE:
                arg.data._to_device()

        for a in self._unique_dat_args:
            kernel.append_arg(a.data.array.data)

        for a in self._all_global_non_reduction_args:
            kernel.append_arg(a.data._array.data)

        for a in self._all_global_reduction_args:
            a.data._allocate_reduction_array(conf['work_group_count'])
            kernel.append_arg(a.data._d_reduc_buffer)

        for cst in Const._definitions():
            kernel.append_arg(cst._array.data)

        for m in self._unique_matrix:
            kernel.append_arg(m._dev_array.data)
            m._upload_array()
            kernel.append_arg(m._dev_rowptr.data)
            kernel.append_arg(m._dev_colidx.data)

        for m in self._matrix_entry_maps:
            m._to_device()
            kernel.append_arg(m._device_values.data)
	print "--> 6 <--"
        if self._is_direct:
            kernel.append_arg(np.int32(self._it_space.size))

            cl.enqueue_nd_range_kernel(_queue, kernel, (conf['thread_count'],), (conf['work_group_size'],), g_times_l=False).wait()
        else:
            kernel.append_arg(np.int32(self._it_space.size))
            kernel.append_arg(self._plan.ind_map.data)
            kernel.append_arg(self._plan.loc_map.data)
            kernel.append_arg(self._plan.ind_sizes.data)
            kernel.append_arg(self._plan.ind_offs.data)
            kernel.append_arg(self._plan.blkmap.data)
            kernel.append_arg(self._plan.offset.data)
            kernel.append_arg(self._plan.nelems.data)
            kernel.append_arg(self._plan.nthrcol.data)
            kernel.append_arg(self._plan.thrcol.data)

            if self._it_space.layers > 1:
	      for arg in self.args:
		if not arg._is_mat and arg._is_vec_map:
		  #print type(arg.map.off)
		  #print self._plan.ind_offs.data
		  arg.map._off_to_device()
		  #myarray = array.to_device(_queue, arg.map.off)
		  #myarray.set(_queue, arg.map.off)
		  kernel.append_arg(arg.map._off_device_values.data)

            block_offset = 0
            print self._plan.ncolors
            print self._plan.ncolblk[0]
            for i in range(self._plan.ncolors):
                blocks_per_grid = int(self._plan.ncolblk[i])
                threads_per_block = min(_max_work_group_size, conf['partition_size'])
                print threads_per_block
                thread_count = threads_per_block * blocks_per_grid

		print block_offset
                kernel.set_last_arg(np.int32(block_offset))
                cl.enqueue_nd_range_kernel(_queue, kernel, (int(thread_count),), (int(threads_per_block),), g_times_l=False).wait()
                block_offset += blocks_per_grid

	print "--> 7 <--"
        # mark !READ data as dirty
        for arg in self.args:
            if arg.access is not READ:
                arg.data.state = DeviceDataMixin.DEVICE
            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

        for mat in [arg.data for arg in self._matrix_args]:
            mat.assemble()
	print "--> 8 <--"
        for a in self._all_global_reduction_args:
            a.data._post_kernel_reduction_task(conf['work_group_count'], a.access)

        if self._has_soa:
            op2stride.remove_from_namespace()

#Monkey patch pyopencl.Kernel for convenience
_original_clKernel = cl.Kernel

class CLKernel (_original_clKernel):
    def __init__(self, *args, **kargs):
        super(CLKernel, self).__init__(*args, **kargs)
        self._karg = 0

    def reset_args(self):
        self._karg = 0;

    def append_arg(self, arg):
        self.set_arg(self._karg, arg)
        self._karg += 1

    def set_last_arg(self, arg):
        self.set_arg(self._karg	, arg)

cl.Kernel = CLKernel

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

def _setup():
    global _ctx
    global _queue
    global _pref_work_group_count
    global _max_local_memory
    global _address_bits
    global _max_work_group_size
    global _has_dpfloat
    global _warpsize
    global _AMD_fixes
    global _reduction_task_cache

    _ctx = cl.create_some_context()
    _queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    _pref_work_group_count = _queue.device.max_compute_units
    _max_local_memory = _queue.device.local_mem_size
    _address_bits = _queue.device.address_bits
    _max_work_group_size = _queue.device.max_work_group_size
    _has_dpfloat = 'cl_khr_fp64' in _queue.device.extensions or 'cl_amd_fp64' in _queue.device.extensions
    if not _has_dpfloat:
        warnings.warn('device does not support double precision floating point computation, expect undefined behavior for double')

    if _queue.device.type == cl.device_type.CPU:
        _warpsize = 1
    elif _queue.device.type == cl.device_type.GPU:
        # assumes nvidia, will probably fail with AMD gpus
        _warpsize = 32

    _AMD_fixes = _queue.device.platform.vendor in ['Advanced Micro Devices, Inc.']
    _reduction_task_cache = dict()

_debug = False
_ctx = None
_queue = None
_pref_work_group_count = 0
_max_local_memory = 0
_address_bits = 32
_max_work_group_size = 0
_has_dpfloat = False
_warpsize = 0
_AMD_fixes = False
_reduction_task_cache = None

_jinja2_env = Environment(loader=PackageLoader("pyop2", "assets"))
_jinja2_direct_loop = _jinja2_env.get_template("opencl_direct_loop.jinja2")
_jinja2_indirect_loop = _jinja2_env.get_template("opencl_indirect_loop.jinja2")
