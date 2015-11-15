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


def test_basics():
    f = Rosenbrock(1, 10)
    assert f.value() == 1.0
    f.make_step([1.0, 1.0])
    assert f.value() == 0.0
    f.make_step([1.0, 1.0])
    assert f.value() == 41.0
    f.step_back()
    assert f.value() == 0.0


def test_gradient():
    fn = Rosenbrock(1, 10)
    xs = [np.random.normal(0, 1, 2) for ix in xrange(100)]
    fn.test_gradient(xs)


def test_hessian():
    fn = Rosenbrock(1, 10)
    xs = [np.random.normal(0, 1, 2) for ix in xrange(100)]
    fn.test_hessian(xs)


def test_dot_hessian():
    fn = Rosenbrock(1, 10)
    xs = [np.random.normal(0, 1, 2) for ix in xrange(100)]
    while True:
        y = np.random.normal(0, 1, 2)
        norm = np.linalg.norm(y)
        if norm > 1e-3:
            y /= norm
            break
    fn.test_dot_hessian(xs, y)
