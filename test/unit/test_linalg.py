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

import pytest
import numpy as np

from pyop2 import op2

nelems = 8

@pytest.fixture
def set():
    return op2.Set(nelems)

@pytest.fixture
def x(set):
    return op2.Dat(set, None, np.float64, "x")

@pytest.fixture
def y(set):
    return op2.Dat(set, np.arange(1,nelems+1), np.float64, "y")

@pytest.fixture
def yi(set):
    return op2.Dat(set, np.arange(1,nelems+1), np.int64, "y")

@pytest.fixture
def x2():
    return op2.Dat(op2.Set(nelems, (1,2)), np.zeros(2*nelems), np.float64, "x")

@pytest.fixture
def y2():
    return op2.Dat(op2.Set(nelems, (2,1)), np.zeros(2*nelems), np.float64, "y")

class TestLinAlgOp:
    """
    Tests of linear algebra operators returning a new Dat.
    """

    def test_add(self, backend, x, y):
        x._data = 2*y.data
        assert all((x+y).data == 3*y.data)

    def test_sub(self, backend, x, y):
        x._data = 2*y.data
        assert all((x-y).data == y.data)

    def test_mul(self, backend, x, y):
        x._data = 2*y.data
        assert all((x*y).data == 2*y.data*y.data)

    def test_div(self, backend, x, y):
        x._data = 2*y.data
        assert all((x/y).data == 2.0)

    def test_add_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 + y2

    def test_sub_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 - y2

    def test_mul_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 * y2

    def test_div_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 / y2

    def test_add_scalar(self, backend, x, y):
        x._data = y.data + 1.0
        assert all(x.data == (y+1.0).data)

    def test_sub_scalar(self, backend, x, y):
        x._data = y.data - 1.0
        assert all(x.data == (y-1.0).data)

    def test_mul_scalar(self, backend, x, y):
        x._data = 2*y.data
        assert all(x.data == (y*2.0).data)

    def test_div_scalar(self, backend, x, y):
        x._data = 2*y.data
        assert all((x/2.0).data == y.data)

    def test_add_ftype(self, backend, y, yi):
        x = y + yi
        assert x.data.dtype == np.float64

    def test_sub_ftype(self, backend, y, yi):
        x = y - yi
        assert x.data.dtype == np.float64

    def test_mul_ftype(self, backend, y, yi):
        x = y * yi
        assert x.data.dtype == np.float64

    def test_div_ftype(self, backend, y, yi):
        x = y / yi
        assert x.data.dtype == np.float64

    def test_add_itype(self, backend, y, yi):
        xi = yi + y
        assert xi.data.dtype == np.int64

    def test_sub_itype(self, backend, y, yi):
        xi = yi - y
        assert xi.data.dtype == np.int64

    def test_mul_itype(self, backend, y, yi):
        xi = yi * y
        assert xi.data.dtype == np.int64

    def test_div_itype(self, backend, y, yi):
        xi = yi / y
        assert xi.data.dtype == np.int64

class TestLinAlgIop:
    """
    Tests of linear algebra operators modifying a Dat in place.
    """

    def test_iadd(self, backend, x, y):
        x._data = 2*y.data
        x += y
        assert all(x.data == 3*y.data)

    def test_isub(self, backend, x, y):
        x._data = 2*y.data
        x -= y
        assert all(x.data == y.data)

    def test_imul(self, backend, x, y):
        x._data = 2*y.data
        x *= y
        assert all(x.data == 2*y.data*y.data)

    def test_idiv(self, backend, x, y):
        x._data = 2*y.data
        x /= y
        assert all(x.data == 2.0)

    def test_iadd_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 += y2

    def test_isub_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 -= y2

    def test_imul_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 *= y2

    def test_idiv_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 /= y2

    def test_iadd_scalar(self, backend, x, y):
        x._data = y.data + 1.0
        y += 1.0
        assert all(x.data == y.data)

    def test_isub_scalar(self, backend, x, y):
        x._data = y.data - 1.0
        y -= 1.0
        assert all(x.data == y.data)

    def test_imul_scalar(self, backend, x, y):
        x._data = 2*y.data
        y *= 2.0
        assert all(x.data == y.data)

    def test_idiv_scalar(self, backend, x, y):
        x._data = 2*y.data
        x /= 2.0
        assert all(x.data == y.data)

    def test_iadd_ftype(self, backend, y, yi):
        y += yi
        assert y.data.dtype == np.float64

    def test_isub_ftype(self, backend, y, yi):
        y -= yi
        assert y.data.dtype == np.float64

    def test_imul_ftype(self, backend, y, yi):
        y *= yi
        assert y.data.dtype == np.float64

    def test_idiv_ftype(self, backend, y, yi):
        y /= yi
        assert y.data.dtype == np.float64

    def test_iadd_itype(self, backend, y, yi):
        yi += y
        assert yi.data.dtype == np.int64

    def test_isub_itype(self, backend, y, yi):
        yi -= y
        assert yi.data.dtype == np.int64

    def test_imul_itype(self, backend, y, yi):
        yi *= y
        assert yi.data.dtype == np.int64

    def test_idiv_itype(self, backend, y, yi):
        yi /= y
        assert yi.data.dtype == np.int64

class TestLinAlgScalar:
    """
    Tests of linear algebra operators return a scalar.
    """

    def test_norm(self, backend):
        n = op2.Dat(op2.Set(2), [3,4], np.float64, "n")
        assert abs(n.norm - 5) < 1e-12
