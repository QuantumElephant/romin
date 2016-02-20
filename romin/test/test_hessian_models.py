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


import numpy as np

from romin import *
from romin.test.random_seed import numpy_random_seed


def test_lbase():
    hm = LBaseHessianModel(3)
    assert hm.backup is None
    assert hm.size == 0
    hm.feed(np.random.normal(0, 1, 4), np.random.normal(0, 1, 4))
    assert hm.backup is None
    assert hm.size == 1
    hm.feed(np.random.normal(0, 1, 4), np.random.normal(0, 1, 4))
    assert hm.backup is None
    assert hm.size == 2
    hm.feed(np.random.normal(0, 1, 4), np.random.normal(0, 1, 4))
    assert hm.backup is None
    assert hm.size == 3
    hm.feed(np.random.normal(0, 1, 4), np.random.normal(0, 1, 4))
    assert hm.backup is not None
    assert hm.size == 3
    hm._restore_backup()
    assert hm.backup is None
    assert hm.size == 3
    hm._restore_backup()
    assert hm.backup is None
    assert hm.size == 2
    hm.reset()
    assert hm.backup is None
    assert hm.size == 0


def test_compare_sr1_lsr1_constant():
    n = 5
    for irep1 in xrange(100):
        with numpy_random_seed(irep1):
            A = np.random.normal(0, 1, (n, n))
            A = (A + A.T)
            hm = SR1HessianModel(eps_skip=0.0)
            lhm = LSR1HessianModel(n, eps_skip=0.0)
            for i in xrange(3):
                delta_x = np.zeros(n)
                delta_x[i] = 1.0
                delta_g = np.dot(A, delta_x)
                hm.feed(delta_x, delta_g)
                lhm.feed(delta_x, delta_g)
            for irep2 in xrange(100):
                v = np.random.normal(0, 1, n)
                o1 = hm.dot_hessian(v)
                o2 = lhm.dot_hessian(v)
                assert abs(o1 - o2).max() < 1e-8


def test_compare_sr1_lsr1_diag():
    n = 5
    for irep1 in xrange(100):
        with numpy_random_seed(irep1):
            A = np.random.normal(0, 1, (n, n))
            A = (A + A.T)
            diag0 = np.random.uniform(1, 2, n)
            hm = SR1HessianModel(np.diag(diag0), eps_skip=0.0)
            lhm = LSR1HessianModel(n, diag0, eps_skip=0.0)
            for i in xrange(3):
                delta_x = np.zeros(n)
                delta_x[i] = 1.0
                delta_g = np.dot(A, delta_x)
                hm.feed(delta_x, delta_g)
                lhm.feed(delta_x, delta_g)
            for irep2 in xrange(100):
                v = np.random.normal(0, 1, n)
                o1 = hm.dot_hessian(v)
                o2 = lhm.dot_hessian(v)
                assert abs(o1 - o2).max() < 1e-8
