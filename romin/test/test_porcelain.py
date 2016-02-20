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


from romin import *

a = 1.0
b = 17.0


def rb_value(x):
    return (a - x[0])**2 + b*(x[0] - x[1]**2)**2


def rb_gradient(x):
    return np.array([
        -2*(a - x[0]) + 2*b*(x[0] - x[1]**2),
        -4*b*(x[0]*x[1] - x[1]**3),
    ])


def rb_hessian(x):
    return np.array([
        [2 + 2*b,
         -4*b*x[1]],
        [-4*b*x[1],
         -4*b*(x[0] - 3*x[1]**2)],
    ])


def rb_dot_hessian(x, y):
    return np.dot(rb_hessian(x), y)


def test_rosenbrock_gradient():
    xs = [np.random.normal(0, 1, 2) for ix in xrange(100)]
    deriv_check(rb_value, rb_gradient, xs)


def test_rosenbrock_hessian():
    xs = [np.random.normal(0, 1, 2) for ix in xrange(100)]
    deriv_check(rb_gradient, rb_hessian, xs)


def test_porcelain_rb_ntr():
    x = minimizer_ntr(np.array([2.0, 5.0]), rb_value, rb_gradient, rb_dot_hessian)
    assert rms(rb_gradient(x)) < 1e-7
    assert abs(x - 1.0).max() < 1e-7


def test_porcelain_rb_qntr():
    x = minimizer_ntr(np.array([2.0, 5.0]), rb_value, rb_gradient)
    assert rms(rb_gradient(x)) < 1e-7
    assert abs(x - 1.0).max() < 1e-7
