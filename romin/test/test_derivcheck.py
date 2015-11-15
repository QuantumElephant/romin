# -*- coding: utf-8 -*-
# RoMin is a robust minimizer.
# Copyright (C) 2011-2015 Toon Verstraelen
#
# This file is part of RoMin.
#
# RoMin is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# RoMin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#--
#pylint: skip-file


from romin import *
from romin.test.random_seed import numpy_random_seed


def check_deriv_check_0d(nx, x_shape):
    f = lambda x: 0.5*np.sum(x**2)
    g = lambda x: x
    xs = [np.random.normal(0, 1, x_shape) for ix in xrange(nx)]
    deriv_check(f, g, xs)


def test_deriv_check_0d():
    yield check_deriv_check_0d, 1, None
    yield check_deriv_check_0d, 1, (10, )
    yield check_deriv_check_0d, 1, (3, 4)
    yield check_deriv_check_0d, 10, None
    yield check_deriv_check_0d, 10, (10, )
    yield check_deriv_check_0d, 10, (3, 4)


def check_deriv_check_nd(nx, x_shape):
    f = lambda x: 0.5*x**2
    def g(x):
        result = np.zeros(x_shape + x_shape)
        for idx, val in np.lib.index_tricks.ndenumerate(x):
            result[idx + idx] = val
        return result
    xs = [np.random.normal(0, 1, x_shape) for ix in xrange(nx)]
    deriv_check(f, g, xs)


def test_deriv_check_nd():
    yield check_deriv_check_nd, 1, (10, )
    yield check_deriv_check_nd, 1, (3, 4)
    yield check_deriv_check_nd, 10, (10, )
    yield check_deriv_check_nd, 10, (3, 4)


def check_deriv_check_extra1(nx):
    f = lambda x: 0.5*(x**2).sum(axis=1)
    def g(x):
        result = np.zeros((4, 4, 3), float)
        for i0 in xrange(4):
            for i1 in xrange(3):
                result[i0, i0, i1] = x[i0, i1]
        return result
    xs = [np.random.normal(0, 1, (4, 3)) for ix in xrange(nx)]
    deriv_check(f, g, xs)


def test_deriv_check_extra1():
    yield check_deriv_check_extra1, 1
    yield check_deriv_check_extra1, 10
