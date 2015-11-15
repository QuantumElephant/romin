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
from romin.deriv_check import deriv_check

__all__ = ['Rosenbrock']


class Objective(object):
    @property
    def dof(self):
        '''The number of degrees of freedom'''
        raise NotImplementedError

    @property
    def hessian_is_approximate(self):
        '''Whether the Hessian is approximate and history-dependent'''
        raise NotImplementedError

    def value(self):
        '''The value of the objective in the current point'''
        raise NotImplementedError

    def gradient(self):
        '''The gradient of the objective in the current point'''
        raise NotImplementedError

    def hessian(self):
        ''''The Hessian of the objective in the current point'''
        raise NotImplementedError

    def dot_hessian(self, y):
        '''The dot product of the Hessian with a test vector in the current point'''
        raise NotImplementedError

    def reset_hessian(self):
        '''Reset the internal state of the history-dependent Hessian, if any'''
        pass

    def make_step(self, delta_x):
        '''Apply the given step to the current point'''
        raise NotImplementedError

    def step_back(self):
        '''Undo the last step, works only once'''
        raise NotImplementedError

    def reset(self, x0):
        '''Reset the point to the given value'''
        raise NotImplementedError

    def test_gradient(self, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3,
                      discard=0.1, verbose=False):
        '''Test the gradient implementation

        See ``romin.deriv_check.deriv_check`` for the documentation of the parameters.
        '''
        def f(x):
            self.reset(x)
            return self.value()
        def g(x):
            self.reset(x)
            return self.gradient()
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, discard, verbose)


    def test_hessian(self, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3,
                     discard=0.1, verbose=False):
        '''Test the hessian implementation

        See ``romin.deriv_check.deriv_check`` for the documentation of the parameters.
        '''
        def f(x):
            self.reset(x)
            return self.gradient()
        def g(x):
            self.reset(x)
            return self.hessian()
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, discard, verbose)


    def test_dot_hessian(self, xs, y, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3,
                         discard=0.1, verbose=False):
        '''Test the dot_hessian implementation

        See ``romin.deriv_check.deriv_check`` for the documentation of the parameters.
        '''
        def f(x):
            self.reset(x)
            return np.dot(self.gradient(), y)
        def g(x):
            self.reset(x)
            return self.dot_hessian(y)
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, discard, verbose)


class Rosenbrock(Objective):
    '''A Rosenbrock function with parameters a and b.

    .. math::

        f(x_0, x_1) = (a - x_0)^2 + b(x_0 - x_1^2)^2

    '''
    def __init__(self, a, b, x=None):
        '''Initialize the Rosenbrock function

        Parameters
        ----------
        a, b : float
               The parameters of the Rosebrock function
        x : np.ndarray, shape = (2,), dtype=float
            Initial point
        '''
        self.a = a
        self.b = b
        if x is None:
            self.x = np.zeros(2, float)
        else:
            self.x = x

    @property
    def dof(self):
        return 2

    @property
    def hessian_is_approximate(self):
        return False

    def value(self):
        return (self.a - self.x[0])**2 + self.b*(self.x[0] - self.x[1]**2)**2

    def gradient(self):
        return np.array([
            -2*(self.a - self.x[0]) + 2*self.b*(self.x[0] - self.x[1]**2),
            -4*self.b*(self.x[0]*self.x[1] - self.x[1]**3),
        ])

    def hessian(self):
        return np.array([
            [2 + 2*self.b,
             -4*self.b*self.x[1]],
            [-4*self.b*self.x[1],
             -4*self.b*(self.x[0] - 3*self.x[1]**2)],
        ])

    def dot_hessian(self, y):
        return np.dot(self.hessian(), y)

    def make_step(self, delta_x):
        self.old_x = self.x.copy()
        self.x += delta_x

    def step_back(self):
        self.x[:] = self.old_x

    def reset(self, x):
        self.old_x = self.x.copy()
        self.x[:] = x
