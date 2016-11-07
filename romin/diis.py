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
"""General-purpose basic DIIS implenetation"""


import numpy as np
from collections import namedtuple


__all__ = ['DIIS']


DIISRecord = namedtuple('DIISRecord', ['x', 'y'])


class DIIS(object):
    """A general-purpose DIIS implementation"""

    def __init__(self, nvec):
        """Initialize a DIIS instance

        Parameters
        ----------
        nvec : int
               The maximum history size. Older vectors are dropped if needed.
        """
        self.nvec = nvec
        self.history = []

    def feed(self, x, y):
        """Add a new record to the DIIS history

        Parameters
        ----------
        x : np.ndarray
            A collection of function arguments.
        y : np.ndarray
            A collection of function values, which you want to zero
        """
        # Store the data.
        self.history.append(DIISRecord(x, y))

        # Drop oldest data when history is getting too long.
        if len(self.history) > self.nvec:
            del self.history[0]

    def guess(self, convex=False, downhill=False, cnmax=1e5, validation=None):
        """Make a guess of x that minimizes the norm of y

        Parameters
        ----------
        convex : bool
                 When True, a non-convex linear combination of previous x_list records
                 is not allowed. If it occurs, the oldest record is removed from the
                 history and a new attempt is made. This is repeated until a convex
                 solution is found or until there is only one record left in the
                 history.
        downhill : bool
                   Only downhill moves are allowed with respect to the last record. In
                   case of a non-downhill move, the history is reduced. This only makes
                   sense if y_list contains the derivatives of an objective towards the
                   elements of x_list.
        cnmax : float
                The maximum allowed condition number of the DIIS matrix. If exceeded,\
                the history is reduced.
        validation : fn(*x_list) -> bool
                     A user provided function that checks the validity of an extrapolated
                     x_list. If invalid, the history is reduced.

        Returns
        -------
        new_x : list of np.ndarray or float
                An interpolation between previous x values in the history that should
                minimize the norm of y.
        """
        def solve():
            """Try to solve the DIIS equations based on the current history"""
            # Build the matrix equation
            size = len(self.history)
            A = np.zeros((size+1, size+1), float)
            B = np.zeros((size+1), float)
            A[size, :size] = 1
            A[:size, size] = 1
            B[size] = 1
            for i0 in xrange(size):
                for i1 in xrange(i0, size):
                    A[i0, i1] = np.dot(self.history[i0].y, self.history[i1].y)
                    A[i1, i0] = A[i0, i1]
            # Rescale the variables
            scale = np.trace(A)/size
            A[:-1,:-1] /= scale
            B[-1] *= scale

            # Check conditioning. If too poor, None is returned.
            evals = abs(np.linalg.eigvalsh(A))
            if evals.min()*cnmax < evals.max():
                return None

            # Solve
            coeffs = np.linalg.solve(A, B)/scale

            # Check solution. If not valid, None is returned
            if convex and (coeffs.min() < 0 or coeffs.max() > 1):
                return None
            new_x = 0.0
            for coeff, record in zip(coeffs, self.history):
                new_x += coeff*record.x
            
            # Downhill check
            if downhill:
                delta_x = new_x - self.history[-1].x
                last_y = self.history[-1].y
                if np.dot(delta_x, last_y) > 0:
                    return None
            
            # User validation
            if (validation is not None) and (not validation(new_x)):
                return None

            # Return solution
            return new_x

        while len(self.history) > 1:
            new_x = solve()
            if new_x is None:
                del self.history[0]
            else:
                return new_x
