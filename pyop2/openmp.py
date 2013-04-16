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

"""OP2 OpenMP backend."""

import os
import numpy as np
import math

from exceptions import *
from utils import *
import op_lib_core as core
import runtime_base as rt
from runtime_base import *
import device

# hard coded value to max openmp threads
_max_threads = 32

class Mat(rt.Mat):
    # This is needed for the test harness to check that two Mats on
    # the same Sparsity share data.
    @property
    def _colidx(self):
        return self._sparsity._colidx

    @property
    def _rowptr(self):
        return self._sparsity._rowptr

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(device.ParLoop):
    def compute(self):
        _fun = self.generate_code()
        _args = [self._it_space.size]
        for arg in self.args:
            if arg._is_mat:
                _args.append(arg.data.handle.handle)
            else:
                _args.append(arg.data.data)

            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    _args.append(map.values)

        for c in Const._definitions():
            _args.append(c.data)



        part_size = self._it_space.partsize  #TODO: compute partition size

        # Create a plan, for colored execution
        if [arg for arg in self.args if arg._is_indirect or arg._is_mat]:
            plan = device.Plan(self._kernel, self._it_space.iterset,
                               *self._unwound_args,
                               partition_size=part_size,
                               matrix_coloring=True)

        else:
            # Create a fake plan for direct loops.
            # Make the fake plan according to the number of cores available
            # to OpenMP
            class FakePlan:
                def __init__(self, iset, part_size):
                    nblocks = int(math.ceil(iset.size / float(part_size)))
                    self.ncolors = 1
                    self.ncolblk = np.array([nblocks], dtype=np.int32)
                    self.blkmap = np.arange(nblocks, dtype=np.int32)
                    self.nelems = np.array([min(part_size, iset.size - i * part_size) for i in range(nblocks)],
                                           dtype=np.int32)

            plan = FakePlan(self._it_space.iterset, part_size)

        _args.append(part_size)
        _args.append(plan.ncolors)
        _args.append(plan.blkmap)
        _args.append(plan.ncolblk)
        _args.append(plan.nelems)

        for arg in self.args:
            if  self._it_space.layers > 1:
              if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                   if map.off != None:
                       _args.append(map.off)

        _fun(*_args)

    def generate_code(self):

        key = self._cache_key
        _fun = device._parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        def c_arg_name(arg):
            name = arg.data.name
            if arg._is_indirect and not (arg._is_vec_map or arg._uses_itspace):
                name += str(arg.idx)
            return name

        def c_vec_name(arg):
            return c_arg_name(arg) + "_vec"

        def c_map_name(arg):
            return c_arg_name(arg) + "_map"

        def c_wrapper_arg(arg):
            val = "PyObject *_%(name)s" % {'name' : c_arg_name(arg) }
            if arg._is_indirect or arg._is_mat:
                val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)}
                maps = as_tuple(arg.map, Map)
                if len(maps) is 2:
                    val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)+'2'}
            return val

        def c_wrapper_dec(arg):
            if arg._is_mat:
                val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                     { "name": c_arg_name(arg) }
            else:
                val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                  {'name' : c_arg_name(arg), 'type' : arg.ctype}
            if arg._is_indirect or arg._is_mat:
                val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                       {'name' : c_map_name(arg)}
            if arg._is_mat:
                val += ";\nint *%(name)s2 = (int *)(((PyArrayObject *)_%(name)s2)->data)" % \
                           {'name' : c_map_name(arg)}
            return val

        def c_ind_data(arg, idx):
            return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'map_name' : c_map_name(arg),
                     'map_dim' : arg.map.dim,
                     'idx' : idx,
                     'dim' : arg.data.cdim}

        def c_kernel_arg(arg):
            if arg._uses_itspace:
                if arg._is_mat:
                    name = "p_%s[tid]" % c_arg_name(arg) # "p_%s[tid]" % c_arg_name(arg)
                    if arg.data._is_vector_field:
                        return name
                    elif arg.data._is_scalar_field:
                        idx = ''.join(["[i_%d]" % i for i, _ in enumerate(arg.data.dims)])
                        return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                            {'t' : arg.ctype,
                             'name' : name,
                             'idx' : idx}
                    else:
                        raise RuntimeError("Don't know how to pass kernel arg %s" % arg)
                else:
                    return c_ind_data(arg, "i_%d" % arg.idx.index)
            elif arg._is_indirect:
                if arg._is_vec_map:
                    return "%s" % c_vec_name(arg) #"%s[tid]" % c_vec_name(arg)
                return c_ind_data(arg, arg.idx)
            elif arg._is_global_reduction:
                return "%(name)s_l1[0]" % { #"%(name)s_l1[tid]" % {
                  'name' : c_arg_name(arg)}
            elif isinstance(arg.data, Global):
                return c_arg_name(arg)
            else:
                return "%(name)s + i * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'dim' : arg.data.cdim}

        def c_vec_dec(arg):
            val = []
            if arg._is_vec_map:
                val.append(";\n%(type)s *%(vec_name)s[%(dim)s]" % #val.append(";\n%(type)s *%(vec_name)s[%(max_threads)s][%(dim)s]"
                       {'type' : arg.ctype,
                        'vec_name' : c_vec_name(arg),
                        'dim' : arg.map.dim,
                        'max_threads': _max_threads})
            return ";\n".join(val)

        def c_vec_init(arg):
            val = []
            for i in range(arg.map._dim):
                val.append("%(vec_name)s[%(idx)s] = %(data)s" % # val.append("%(vec_name)s[tid][%(idx)s] = %(data)s" %
                           {'vec_name' : c_vec_name(arg),
                            'idx' : i,
                            'data' : c_ind_data(arg, i)} )
            return ";\n".join(val)

        def c_addto_scalar_field(arg):
            name = c_arg_name(arg)
            p_data = 'p_%s[tid]' % name
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim

            return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
                {'mat' : name,
                 'vals' : p_data,
                 'nrows' : nrows,
                 'ncols' : ncols,
                 'rows' : "%s + i * %s" % (c_map_name(arg), nrows),
                 'cols' : "%s2 + i * %s" % (c_map_name(arg), ncols),
                 'insert' : arg.access == rt.WRITE }

        def c_addto_vector_field(arg):
            name = c_arg_name(arg)
            p_data = 'p_%s[tid]' % name
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim
            dims = arg.data.sparsity.dims
            rmult = dims[0]
            cmult = dims[1]
            s = []
            for i in xrange(rmult):
                for j in xrange(cmult):
                    idx = '[%d][%d]' % (i, j)
                    val = "&%s%s" % (p_data, idx)
                    row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                          {'m' : rmult,
                           'map' : c_map_name(arg),
                           'dim' : nrows,
                           'i' : i }
                    col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                          {'m' : cmult,
                           'map' : c_map_name(arg),
                           'dim' : ncols,
                           'j' : j }

                    s.append('addto_scalar(%s, %s, %s, %s, %d)' \
                            % (name, val, row, col, arg.access == rt.WRITE))
            return ';\n'.join(s)

        def c_assemble(arg):
            name = c_arg_name(arg)
            return "assemble_mat(%s)" % name

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d){" % (i, i, d, i)

        def tmp_decl(arg, extents):
            t = arg.data.ctype
            if arg.data._is_scalar_field:
                dims = ''.join(["[%d]" % d for d in extents])
            elif arg.data._is_vector_field:
                dims = ''.join(["[%d]" % d for d in arg.data.dims])
            else:
                raise RuntimeError("Don't know how to declare temp array for %s" % arg)
            return "%s p_%s[%s]%s" % (t, c_arg_name(arg), _max_threads, dims)

        def c_zero_tmp(arg):
            name = "p_" + c_arg_name(arg)
            t = arg.ctype
            if arg.data._is_scalar_field:
                idx = ''.join(["[i_%d]" % i for i,_ in enumerate(arg.data.dims)])
                return "%(name)s[tid]%(idx)s = (%(t)s)0" % \
                    {'name' : name, 't' : t, 'idx' : idx}
            elif arg.data._is_vector_field:
                size = np.prod(arg.data.dims)
                return "memset(%(name)s[tid], 0, sizeof(%(t)s) * %(size)s)" % \
                    {'name' : name, 't' : t, 'size' : size}
            else:
                raise RuntimeError("Don't know how to zero temp array for %s" % arg)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])

        def c_reduction_dec(arg):
            return "%(type)s %(name)s_l[%(max_threads)s][%(dim)s]" % \
              {'type' : arg.ctype,
               'name' : c_arg_name(arg),
               'dim' : arg.data.cdim,
               # Ensure different threads are on different cache lines
               'max_threads' : _max_threads}

        def c_reduction_init(arg):
            if arg.access == INC:
                init = "(%(type)s)0" % {'type' : arg.ctype}
            else:
                init = "%(name)s[i]" % {'name' : c_arg_name(arg)}
            return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l[tid][i] = %(init)s" % \
              {'dim' : arg.data.cdim,
               'name' : c_arg_name(arg),
               'init' : init}

        def c_reduction_finalisation(arg):
            d = {'gbl': c_arg_name(arg),
                 'local': "%s_l[thread][i]" % c_arg_name(arg)}
            if arg.access == INC:
                combine = "%(gbl)s[i] += %(local)s" % d
            elif arg.access == MIN:
                combine = "%(gbl)s[i] = %(gbl)s[i] < %(local)s ? %(gbl)s[i] : %(local)s" % d
            elif arg.access == MAX:
                combine = "%(gbl)s[i] = %(gbl)s[i] > %(local)s ? %(gbl)s[i] : %(local)s" % d
            return """
            for ( int thread = 0; thread < nthread; thread++ ) {
                for ( int i = 0; i < %(dim)s; i++ ) %(combine)s;
            }""" % {'combine' : combine,
                    'dim' : arg.data.cdim}

        def c_interm_globals_decl(arg):
            return "%(type)s %(name)s_l1[1][1]" % \
              {'type' : arg.ctype,
               'name' : c_arg_name(arg)
              }

        def c_interm_globals_init(arg):
            return "%(name)s_l1[0][0] = (double)0" % \
              {
               'name' : c_arg_name(arg)
              }

        def c_interm_globals_writeback(arg):
            return "%(name)s_l[tid][0] = %(name)s_l1[0][0]" % \
              {
               'name' : c_arg_name(arg)
              }

        def extrusion_loop(d):
            return "for (int j_0=0; j_0<%d; ++j_0){" % d

        def c_off_init(c):
            return "PyObject *off%(name)s" % {'name' : c }

        def c_off_decl(count):
            return 'int * _off%(cnt)s = (int *)(((PyArrayObject *)off%(cnt)s)->data)' % { 'cnt' : count }

        def c_add_off(c,layers,count):
            return """
            for(int j=0; j<%(layers)s;j++){
                %(name)s[j] += _off%(num)s[j];
            }""" % {'name' : c_vec_name(c),
            'layers' : layers,
            'num' : count}

        def c_rewrite_dats(arg):
            return """
            double * %(name)s_mp = (double *)malloc(%(size)s*sizeof(double));
            """ % {
                'name' : c_arg_name(arg),
                'size' : arg.data.dat_size
                }


        def c_rewrite_data_parallel(arg):
            return """
            #pragma omp parallel for default(shared)
            for(int i = 0; i < %(size)s; i++){
                %(name)s_mp[i] =  %(name)s[i];
            }
            """ %{
                'name' : c_arg_name(arg),
                'size' : arg.data.dat_size
                }

        args = self.args
        _wrapper_args = ', '.join([c_wrapper_arg(arg) for arg in args])

        _tmp_decs = ';\n'.join([tmp_decl(arg, self._it_space.extents) for arg in args if arg._is_mat])
        _wrapper_decs = ';\n'.join([c_wrapper_dec(arg) for arg in args])

        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        _kernel_user_args = [c_kernel_arg(arg) for arg in args]
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_decs = ';\n'.join([c_vec_dec(arg) for arg in args \
                                 if not arg._is_mat and arg._is_vec_map])
        _vec_inits = ';\n'.join([c_vec_init(arg) for arg in args \
                                 if not arg._is_mat and arg._is_vec_map])

        _itspace_loops = '\n'.join([itspace_loop(i,e) for i, e in zip(range(len(self._it_space.extents)), self._it_space.extents)])
        _itspace_loop_close = '}'*len(self._it_space.extents)

        _addtos_vector_field = ';\n'.join([c_addto_vector_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_vector_field])
        _addtos_scalar_field = ';\n'.join([c_addto_scalar_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_scalar_field])

        _assembles = ';\n'.join([c_assemble(arg) for arg in args if arg._is_mat])

        _zero_tmps = ';\n'.join([c_zero_tmp(arg) for arg in args if arg._is_mat])

        _set_size_wrapper = 'PyObject *_%(set)s_size' % {'set' : self._it_space.name}
        _set_size_dec = 'int %(set)s_size = (int)PyInt_AsLong(_%(set)s_size);' % {'set' : self._it_space.name}
        _set_size = '%(set)s_size' % {'set' : self._it_space.name}

        _reduction_decs = ';\n'.join([c_reduction_dec(arg) for arg in args if arg._is_global_reduction])
        _reduction_inits = ';\n'.join([c_reduction_init(arg) for arg in args if arg._is_global_reduction])
        _reduction_finalisations = '\n'.join([c_reduction_finalisation(arg) for arg in args if arg._is_global_reduction])

        _interm_globals_decl = ';\n'.join([c_interm_globals_decl(arg) for arg in args if arg._is_global_reduction])
        _interm_globals_init = ';\n'.join([c_interm_globals_init(arg) for arg in args if arg._is_global_reduction])
        _interm_globals_writeback = ';\n'.join([c_interm_globals_writeback(arg) for arg in args if arg._is_global_reduction])

        _rewrite_dats = ';\n'.join([c_rewrite_dats(arg) for arg in args if not arg._is_mat and arg._is_vec_map])

        _rewrite_data_parallel = ';\n'.join([c_rewrite_data_parallel(arg) for arg in args if not arg._is_mat and arg._is_vec_map])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        count = 0
        _dd = "%d"
        _ff = "%f"
        off_i = []
        off_d = []
        off_a = []
        _off_args = ""
        _off_inits = ""
        _apply_offset = ""
        _extr_loop = ""
        _extr_loop_close = ""
        if self._it_space.layers > 1:
            for arg in args:
                if not arg._is_mat and arg._is_vec_map:
                    count += 1
                    off_i.append(c_off_init(count))
                    off_d.append(c_off_decl(count))
                    off_a.append(c_add_off(arg,arg.map.off.size,count))
            if off_i != []:
              _off_args = ', '
              _off_args +=', '.join(off_i)
              _off_inits = ';\n'.join(off_d)
              _apply_offset = ' \n'.join(off_a)
              _extr_loop = '\n'
              _extr_loop += extrusion_loop(self._it_space.layers-1)
              _extr_loop_close = '}'
              _kernel_args += ', j_0'

        wrapper = """
            void wrap_%(kernel_name)s__(%(set_size_wrapper)s, %(wrapper_args)s %(const_args)s, PyObject* _part_size, PyObject* _ncolors, PyObject* _blkmap, PyObject* _ncolblk, PyObject* _nelems %(off_args)s) {

            int part_size = (int)PyInt_AsLong(_part_size);
            int ncolors = (int)PyInt_AsLong(_ncolors);
            int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
            int* ncolblk = (int *)(((PyArrayObject *)_ncolblk)->data);
            int* nelems = (int *)(((PyArrayObject *)_nelems)->data);

            %(set_size_dec)s;
            %(wrapper_decs)s;
            %(const_inits)s;

            %(tmp_decs)s;
            %(off_inits)s;

            //printf("-1\\n");
            #ifdef _OPENMP
            int nthread = omp_get_max_threads();
            #else
            int nthread = 1;
            #endif

            //%(reduction_decs)s;

            double g_l[32][1024];

            #pragma omp parallel // default(shared)
            {
              int tid = omp_get_thread_num();
              //%(reduction_inits)s;
              for(int i=0;i<1024;i++) g_l[tid][i]=0;
            }

            if (likwid_init == 0){
                likwid_markerInit();
            }
            int boffset = 0;
            int __b,tid;
            int lim;

            for ( int __col  = 0; __col < ncolors; __col++ ) {
              int nblocks = ncolblk[__col];

              //printf("Colour %(dd)s has %(dd)s partitions\\n", __col, nblocks);
              #pragma omp parallel private(__b,tid, lim) shared(boffset, nblocks, nelems, blkmap, part_size)
              {
                likwid_markerThreadInit();
                likwid_markerStartRegion("accumulate");

                tid = omp_get_thread_num();
                //%(interm_globals_decl)s;
                //%(interm_globals_init)s;
                lim = boffset + nblocks;

                double g_l1[32][1];

                for(int i=0;i<1;i++) {
                    g_l1[tid][i]=0;
                }

                #pragma omp for schedule(static)
                for (__b = boffset; __b < lim; __b++ )
                  {
                  %(vec_decs)s;
                  int bid = blkmap[__b];
                  int nelem = nelems[bid];
                  int efirst = bid * part_size;
                  int lim2 = nelem + efirst;
                  //printf("Thread %(dd)s on cpu %(dd)s \\n", tid, sched_getcpu());

                  for (int i = efirst; i < lim2; i++) {
                    //printf("%(dd)s %(dd)s %(dd)s %(dd)s %(dd)s %(dd)s \\n",coords_map[i*6+0],coords_map[i*6+1], coords_map[i*6+2],coords_map[i*6+3],coords_map[i*6+4],coords_map[i*6+5]);

                    %(vec_inits)s;
                    %(itspace_loops)s
                    %(extr_loop)s
                    %(zero_tmps)s;
                    //%(kernel_name)s(%(kernel_args)s);
                    comp_vol(g_l1[tid], coords_vec, field_vec, j_0);




                    %(addtos_vector_field)s;
                    //colours_vec[0][0] = __col;
                    %(apply_offset)s
                    %(extr_loop_close)s
                    %(itspace_loop_close)s
                    %(addtos_scalar_field)s;

                    //if (bid < 10)
                    //    colours_vec[0][0] = bid;
                    //else
                    //    colours_vec[0][0] = 10;
                    //colours_vec[0][0] = __col;
                  }

                  //getchar();

                  //%(interm_globals_writeback)s;
                }
                //%(interm_globals_writeback)s;
                g_l[tid][0] = g_l1[tid][0];
                //printf("Result: %(ff)s out of %(ff)s\\n", g_l[tid][0], g[0]);
                likwid_markerStopRegion("accumulate");
              }
              %(reduction_finalisations)s
              boffset += nblocks;
            }
            %(assembles)s;
            ;

            if (likwid_init == 99){
                likwid_markerClose();
            }
            likwid_init++;
          }"""

        if any(arg._is_soa for arg in args):
            kernel_code = """
            #include <likwid.h>
            #include <math.h>
            #define OP2_STRIDE(a, idx) a[idx]
            static int likwid_init = 0;
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            #include <likwid.h>
            #include <math.h>
            static int likwid_init = 0;
            inline %(code)s
            """ % {'code' : self._kernel.code }
        code_to_compile =  wrapper % { 'kernel_name' : self._kernel.name,
                                       'wrapper_args' : _wrapper_args,
                                       'wrapper_decs' : _wrapper_decs,
                                       'const_args' : _const_args,
                                       'const_inits' : _const_inits,
                                       'tmp_decs' : _tmp_decs,
                                       'set_size' : _set_size,
                                       'set_size_dec' : _set_size_dec,
                                       'set_size_wrapper' : _set_size_wrapper,
                                       'itspace_loops' : _itspace_loops,
                                       'itspace_loop_close' : _itspace_loop_close,
                                       'vec_inits' : _vec_inits,
                                       'vec_decs' : _vec_decs,
                                       'zero_tmps' : _zero_tmps,
                                       'kernel_args' : _kernel_args,
                                       'addtos_vector_field' : _addtos_vector_field,
                                       'addtos_scalar_field' : _addtos_scalar_field,
                                       'assembles' : _assembles,
                                       'reduction_decs' : _reduction_decs,
                                       'reduction_inits' : _reduction_inits,
                                       'reduction_finalisations' : _reduction_finalisations,
                                       'apply_offset' : _apply_offset,
                                       'off_args' : _off_args,
                                       'off_inits' : _off_inits,
                                       'extr_loop' : _extr_loop,
                                       'extr_loop_close' : _extr_loop_close,
                                       'interm_globals_decl' : _interm_globals_decl,
                                       'interm_globals_init' : _interm_globals_init,
                                       'interm_globals_writeback' : _interm_globals_writeback,
                                       #'rewrite_dats' : _rewrite_dats,
                                       #'rewrite_data_parallel' : _rewrite_data_parallel,
                                       'dd' : _dd,
                                       'ff' : _ff}

        # We need to build with mpicc since that's required by PETSc
        code_to_compile="""
        void wrap_comp_vol__(PyObject *_elements_size, PyObject *_g, PyObject *_coords, PyObject *_coords_map, PyObject *_field, PyObject *_field_map , PyObject* _part_size, PyObject* _ncolors, PyObject* _blkmap, PyObject* _ncolblk, PyObject* _nelems , PyObject *off1, PyObject *off2) {

            int part_size = (int)PyInt_AsLong(_part_size);
            int ncolors = (int)PyInt_AsLong(_ncolors);
            int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
            int* ncolblk = (int *)(((PyArrayObject *)_ncolblk)->data);
            int* nelems = (int *)(((PyArrayObject *)_nelems)->data);

            int elements_size = (int)PyInt_AsLong(_elements_size);;
            double *g = (double *)(((PyArrayObject *)_g)->data);
            double *coords = (double *)(((PyArrayObject *)_coords)->data);
            int *coords_map = (int *)(((PyArrayObject *)_coords_map)->data);
            double *field = (double *)(((PyArrayObject *)_field)->data);
            int *field_map = (int *)(((PyArrayObject *)_field_map)->data);
            ;

            ;
            int * _off1 = (int *)(((PyArrayObject *)off1)->data);
            int * _off2 = (int *)(((PyArrayObject *)off2)->data);

            #ifdef _OPENMP
            int nthread = omp_get_max_threads();
            #else
            int nthread = 1;
            #endif

            double g_l[32][1024];

            #pragma omp parallel // default(shared)
            {
              int tid = omp_get_thread_num();
              for(int i=0;i<1024;i++) g_l[tid][i]=0.0;
            }

            if (likwid_init == 0){
                likwid_markerInit();
            }
            int boffset = 0;
            int __b,tid;
            int lim;

            for ( int __col  = 0; __col < ncolors; __col++ ) {
              int nblocks = ncolblk[__col];

              //printf("Colour %d has %d partitions\\n", __col, nblocks);
              #pragma omp parallel private(__b,tid, lim) shared(boffset, nblocks, nelems, blkmap, part_size)
              {
                likwid_markerThreadInit();
                likwid_markerStartRegion("accumulate");

                tid = omp_get_thread_num();
                double g_l1[1][1];
                g_l1[0][0] = (double)0;
                lim = boffset + nblocks;

                //double g_l1[32][1];

                //for(int i=0;i<1;i++) {
                //    g_l1[tid][i]=0;
                //}

                #pragma omp for schedule(static)
                for (__b = boffset; __b < lim; __b++ )
                  {
                  ;
                  //double c_vec[6][200];
                  //double f_vec[1][100];
                  double *coords_vec[6];
                  double *field_vec[1];

                  int bid = blkmap[__b];
                  int nelem = nelems[bid];
                  int efirst = bid * part_size;
                  int lim2 = nelem + efirst;
                  //printf("Thread %d on cpu %d \\n", tid, sched_getcpu());

                  for (int i = efirst; i < lim2; i++) {
                    //printf("%d %d %d %d %d %d \\n",coords_map[i*6+0],coords_map[i*6+1], coords_map[i*6+2],coords_map[i*6+3],coords_map[i*6+4],coords_map[i*6+5]);


                    coords_vec[0] = coords + coords_map[i * 6 + 0] * 1;
                    coords_vec[1] = coords + coords_map[i * 6 + 1] * 1;
                    coords_vec[2] = coords + coords_map[i * 6 + 2] * 1;
                    coords_vec[3] = coords + coords_map[i * 6 + 3] * 1;
                    coords_vec[4] = coords + coords_map[i * 6 + 4] * 1;
                    coords_vec[5] = coords + coords_map[i * 6 + 5] * 1;
                    field_vec[0] = field + field_map[i * 1 + 0] * 1;

                    //memcpy(coords + coords_map[i * 6 + 0], c_vec[0], 1600);
                    //memcpy(coords + coords_map[i * 6 + 1], c_vec[1], 1600);
                    //memcpy(coords + coords_map[i * 6 + 2], c_vec[2], 1600);
                    //memcpy(coords + coords_map[i * 6 + 3], c_vec[3], 1600);
                    //memcpy(coords + coords_map[i * 6 + 4], c_vec[4], 1600);
                    //memcpy(coords + coords_map[i * 6 + 5], c_vec[5], 1600);
                    //memcpy(field + field_map[i * 1 + 0],   f_vec[0], 800);

                    //printf("",c_vec[0][0]);

                    //coords_vec[0] = &c_vec[0][0];
                    //coords_vec[1] = &c_vec[1][1];
                    //coords_vec[2] = &c_vec[2][2];
                    //coords_vec[3] = &c_vec[3][3];
                    //coords_vec[4] = &c_vec[4][4];
                    //coords_vec[5] = &c_vec[5][5];
                    //field_vec[0] = &f_vec[0][0];

                    for (int j_0=0; j_0<100; ++j_0){
                        ;
                        //comp_vol(g_l1[0], coords_vec, field_vec, j_0);
                        comp_vol(g_l1[0], coords_vec, field_vec, j_0);


                        for(int j=0; j<6;j++){
                            coords_vec[j] += _off1[j];
                        }

                        for(int j=0; j<1;j++){
                            field_vec[j] += _off2[j];
                        }
                    }
                  }
                  //getchar();
                  //g_l[tid][0] = g_l1[0][0];
                }
                g_l[tid][0] = g_l1[0][0];
                //memcpy(g_l[tid], g_l1[tid], 8);
                //printf("Result: %f out of %f\\n", g_l[tid][0], g[0]);
                likwid_markerStopRegion("accumulate");
              }

              for ( int thread = 0; thread < nthread; thread++ ) {
                for ( int i = 0; i < 1; i++ ) g[i] += g_l[thread][i];
              }
              boffset += nblocks;
            }


            if (likwid_init == 99){
                likwid_markerClose();
            }
            likwid_init++;
          }
        """
        #print code_to_compile
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 include_dirs=[OP2_INC, get_petsc_dir()+'/include','/usr/local/include/'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 library_dirs=[OP2_LIB, get_petsc_dir()+'/lib', '/usr/local/lib/'],
                                 libraries=['op2_seq', 'petsc', 'likwid'],
                                 sources=["mat_utils.cxx"],
                                 cppargs=['-fopenmp', '-pthread', '-fprefetch-loop-arrays'],
                                 system_headers=['omp.h'],
                                 lddargs=['-fopenmp', '-pthread'])

        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')

        device._parloop_cache[key] = _fun
        return _fun

def _setup():
    pass
