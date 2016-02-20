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


__all__ = ['Objective', 'HessianModelWrapper']


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

    def make_step(self, delta_x):
        '''Apply the given step to the current point'''
        raise NotImplementedError

    def step_back(self):
        '''Undo the last step, works only once'''
        raise NotImplementedError

    def reset(self, x0):
        '''Reset the point to the given value'''
        raise NotImplementedError

    def reset_hessian(self):
        '''Reset the internal state of the history-dependent Hessian, if any'''
        pass

    def test_gradient(self, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, weights=1,
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
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, weights, discard, verbose)


    def test_hessian(self, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, weights=1,
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
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, weights, discard, verbose)


    def test_dot_hessian(self, xs, y, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3,
                         weights=1, discard=0.1, verbose=False):
        '''Test the dot_hessian implementation

        See ``romin.deriv_check.deriv_check`` for the documentation of the parameters.
        '''
        def f(x):
            self.reset(x)
            return np.dot(self.gradient(), y)
        def g(x):
            self.reset(x)
            return self.dot_hessian(y)
        deriv_check(f, g, xs, eps_x, order, nrep, rel_ftol, weights, discard, verbose)


class HessianModelWrapper(Objective):
    def __init__(self, objective, hessian_model):
        """Adds a Hessian model to an objective without Hessian implementation.

        Parameters
        ----------
        objective : Objective
                    The original objective for which only the value and the gradient are
                    implemented.
        hessian_model : SR1HessianModel, LSR1HessianModel
                        A model to build up an approximate Hessian during the
                        optimization.
        """
        self.objective = objective
        self.hessian_model = hessian_model
        self.last_gradient = None
        self.last_delta_x = None

    @property
    def dof(self):
        '''The number of degrees of freedom'''
        return self.objective.dof

    @property
    def hessian_is_approximate(self):
        '''Whether the Hessian is approximate and history-dependent'''
        return True

    def value(self):
        '''The value of the objective in the current point'''
        return self.objective.value()

    def gradient(self):
        '''The gradient of the objective in the current point'''
        gradient = self.objective.gradient()
        if self.last_delta_x is not None:
            self.hessian_model.feed(self.last_delta_x, gradient-self.last_gradient)
            self.last_delta_x = None
        self.last_gradient = gradient
        return gradient

    def hessian(self):
        ''''The Hessian of the objective in the current point'''
        if hasattr(self.hessian_model, 'hessian'):
            return self.hessian_model.hessian
        else:
            raise NotImplementedError

    def dot_hessian(self, y):
        '''The dot product of the Hessian with a test vector in the current point'''
        return self.hessian_model.dot_hessian(y)

    def make_step(self, delta_x):
        '''Apply the given step to the current point'''
        if self.last_delta_x is None:
            self.last_delta_x = delta_x
        else:
            raise ValueError('make_step called twice without gradient call in between.')
        self.objective.make_step(delta_x)

    def step_back(self):
        '''Undo the last step, works only once'''
        self.objective.step_back()

    def reset(self, x0):
        '''Reset the point to the given value'''
        self.objective.reset(x0)

    def reset_hessian(self):
        '''Reset the internal state of the history-dependent Hessian, if any'''
        self.hessian_model.reset()

    def test_hessian(self, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, weights=1,
                     discard=0.1, verbose=False):
        raise NotImplementedError

    def test_dot_hessian(self, xs, y, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3,
                         weights=1, discard=0.1, verbose=False):
        raise NotImplementedError
