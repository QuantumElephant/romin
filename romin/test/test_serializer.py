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


def test_serializer():
    serializer = Serializer()
    a1 = np.array([1.0, 3.0, 3.0])
    b1 = 2.5
    c1 = np.array([[2.0, 3.0], [0.5, 0.3]])
    x = serializer(a1, b1, c1)
    assert (x == [1.0, 3.0, 3.0, 2.5, 2.0, 3.0, 0.5, 0.3]).all()
    assert len(serializer.shapes) == 3
    assert serializer.shapes[0] == (3,)
    assert serializer.shapes[1] == ()
    assert serializer.shapes[2] == (2, 2)
    a2, b2, c2 = serializer.undo(np.arange(8.0))
    assert (a2 == [0.0, 1.0, 2.0]).all()
    assert b2 == 3.0
    assert (c2 == [[4.0, 5.0], [6.0, 7.0]]).all()
