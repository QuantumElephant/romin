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
from collections import namedtuple


__all__ = ['BaseHessianModel', 'SR1HessianModel', 'LBaseHessianModel', 'LSR1HessianModel']


HessianRecord = namedtuple('HessianRecord', ['delta_x', 'delta_g'])


class BaseHessianModel(object):
    def __init__(self, hessian0=1.0):
        self.hessian0 = hessian0
        self.hessian = hessian0

    def reset(self):
        """Reset the Hessian to the initial value"""
        self.hessian = self.hessian0

    def feed(self, delta_x, delta_g):
        raise NotImplementedError

    def dot_hessian(self, v):
        return np.dot(self.hessian, v)


class SR1HessianModel(BaseHessianModel):
    def __init__(self, hessian0=1.0, eps_skip=1e-8):
        self.eps_skip = eps_skip
        BaseHessianModel.__init__(self, hessian0)

    def feed(self, delta_x, delta_g):
        if isinstance(self.hessian, float):
            self.hessian = np.identity(len(delta_x), float)*self.hessian
        residual = delta_g - np.dot(self.hessian, delta_x)
        denominator = np.dot(residual, delta_x)
        threshold = max(1, np.linalg.norm(residual)*np.linalg.norm(delta_x))*self.eps_skip
        if abs(denominator) > threshold:
            self.hessian += np.outer(residual, residual)/np.dot(residual, delta_x)


class LBaseHessianModel(object):
    """Base class for all low-memory Hessian models

    The base functionality is the storage and management of previous steps and
    corresponding gradient changes.
    """

    def __init__(self, maxsize, diag0=1.0):
        """Initialize the BaseHessianModel

        Parameters
        ----------
        maxsize : int
                  The maximum number of vectors stored
        diag0 : float or np.ndarray, shape=(n,)
                The diagonal of the initial Hessian
        """
        self.maxsize = maxsize
        self.diag0 = diag0
        self.reset()

    @property
    def size(self):
        """The current size of the history"""
        return len(self.history)

    def reset(self):
        """Drop the history"""
        self.history = []
        self.backup = None

    def feed(self, delta_x, delta_g):
        """Store a new couple of step and gradient change

        Parameters
        ----------
        delta_x : np.ndarray, shape=(n,)
                  The change in the argument (of the function for which the Hessian is to
                  be approximated). This is often denoted as s in the literature.
        delta_g : np.ndarray, shape=(n,)
                  The corresponding change in the gradient
        """
        if len(delta_x.shape) != 1:
            raise TypeError('The argument delta_x must be a vector.')
        if len(delta_x) < self.maxsize:
            raise TypeError('The history size is too large for the dimension of delta_x.')
        if delta_x.shape != delta_g.shape:
            raise TypeError('The arguments delta_x and delta_g must have the same shape.')
        self.history.append(HessianRecord(delta_x, delta_g))
        if len(self.history) > self.maxsize:
            self.backup = self.history.pop(0)

    def _restore_backup(self):
        """Restore the history to the point before the last call to self.feed"""
        if self.backup is not None:
            self.history.insert(0, self.backup)
            self.backup = None
        del self.history[-1]

    def dot_hessian(self, v):
        """Return the dot product of the approximate hessian and a test vector v"""
        raise NotImplementedError


class LSR1HessianModel(LBaseHessianModel):
    """L-SR1 update implementation

    See: 10.1007/BF01582063
    Mathematical Programming
    January 1994, Volume 63, Issue 1, pp 129-156
    Representations of quasi-Newton matrices and their use in limited memory methods
    Richard H. Byrd, Jorge Nocedal, Robert B. Schnabel
    """
    def __init__(self, nvec, diag0=1.0, eps_skip=1e-8):
        """Initialize the BaseHessianModel

        Parameters
        ----------
        maxsize : int
                  The maximum number of vectors stored
        diag0 : float or np.ndarray, shape=(n,)
                The diagonal of the initial Hessian
        eps_skip : float
                   If an update would cause the condition number of the small Hessian to
                   go below this threshold, the update is discarded.
        """
        self.eps_skip = eps_skip
        LBaseHessianModel.__init__(self, nvec, diag0)

    def reset(self):
        """Drop the history"""
        LBaseHessianModel.reset(self)
        self.minv = None
        self.basis = None

    def feed(self, delta_x, delta_g):
        """Store a new couple of step and gradient change

        Parameters
        ----------
        delta_x : np.ndarray, shape=(n,)
                  The change in the argument (of the function for which the Hessian is to
                  be approximated). This is often denoted as s in the literature.
        delta_g : np.ndarray, shape=(n,)
                  The corresponding change in the gradient
        """
        LBaseHessianModel.feed(self, delta_x, delta_g)

        # Compute derived quantities but do not store them yet.
        minv = np.zeros((self.size, self.size), float)
        for i in xrange(self.size):
            for j in xrange(i, self.size):
                minv[i, j] = np.dot(self.history[i].delta_x, self.history[j].delta_g) - \
                             np.dot(self.history[i].delta_x, self.diag0*self.history[j].delta_x)
                if i < j:
                    minv[j, i] = minv[i, j]

        # Check if the new Hessian approximation is going to be singular.
        absevals = abs(np.linalg.eigvalsh(minv))
        if (absevals.max() < self.eps_skip) or (absevals.max()*self.eps_skip > absevals.min()):
            self._restore_backup()
            return

        # Continue setting up the new hessian
        self.m = np.linalg.inv(minv)
        self.basis = np.array([delta_g - self.diag0*delta_x for delta_x, delta_g in self.history])

    def dot_hessian(self, v):
        """Return the dot product of the approximate hessian and a test vector v"""
        result = self.diag0*v
        if self.size > 0:
            result += np.dot(self.basis.T, np.dot(self.m, np.dot(self.basis, v)))
        return result
