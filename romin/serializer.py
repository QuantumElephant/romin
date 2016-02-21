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
"""Tool for (un)serializing multiple array arguments"""


import numpy as np


__all__ = ['Serializer']


class Serializer(object):
    def __init__(self):
        self.shapes = None
        self.dof = None

    def _derive_shapes(self, x_list):
        """Derive the shapes of the elements of x_list for later validation."""
        self.shapes = []
        self.dof = 0
        for x in x_list:
            if isinstance(x, float):
                self.shapes.append(())
                self.dof += 1
            else:
                if not isinstance(x, np.ndarray):
                    raise TypeError('Elements of x_list should be float or array.')
                self.shapes.append(x.shape)
                self.dof += np.product(x.shape)

    def _validate_shapes(self, arg_list):
        """Validate the type of x_list and its elements."""
        if len(arg_list) != len(self.shapes):
            raise TypeError('Wrong number of arguments for serialize.')
        for i, shape in enumerate(self.shapes):
            if len(shape) == 0:
                if not isinstance(arg_list[i], float):
                    raise TypeError('Element %i in serialize should be a float.' % i)
            else:
                if not isinstance(arg_list[i], np.ndarray):
                    raise TypeError('Element %i in serialize should be a numpy array.' % i)
                if arg_list[i].shape != shape:
                    raise TypeError('Element %i in serialize should have shape %s.' % (i, shape))

    def __call__(self, *arg_list):
        """Convert an x_list to a long numpy vector."""
        if self.shapes is None:
            self._derive_shapes(arg_list)
        else:
            self._validate_shapes(arg_list)
        result = []
        for arg in arg_list:
            if isinstance(arg, float):
                result.append(arg)
            else:
                result.extend(arg.ravel())
        return np.array(result)

    def undo(self, x):
        """Convert a long vector back to an x_list."""
        if self.shapes is None:
            raise ValueError('Shapes are not known yet.')
        if x.shape != (self.dof,):
            raise TypeError('The argument x does not have the right shape.')
        x_list = []
        begin = 0
        for shape in self.shapes:
            if len(shape) == 0:
                x_list.append(x[begin])
                begin += 1
            else:
                end = begin + np.product(shape)
                x_list.append(x[begin:end].reshape(shape))
                begin = end
        return x_list
