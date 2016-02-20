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
from romin.test.test_objectives import Rosenbrock


def test_min_ntr_rosenbrock():
    for b in xrange(3, 100, 10):
        fn = Rosenbrock(1, b, np.array([2.0, 5.0]))
        minimize_objective_ntr(fn)
        assert abs(fn.gradient()).max() < 1e-7
        assert abs(fn.x - 1.0).max() < 1e-7
