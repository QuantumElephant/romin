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
from romin import Objective
from romin.test.random_seed import numpy_random_seed


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
            self.x = np.array(x)

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


class NobleAtoms(Objective):
    '''The potential energy of a (small) number of noble gas atoms'''
    def __init__(self, epsilon, sigma, natom, x):
        '''Initialize the potential energy of the noble gass atoms

        Parameters
        ----------
        epsilon, sigma : float
                         Lennard-Jones parameters
        natom : int
                The number of atoms
        x : np.ndarray, shape = (3*natom,), dtype=float
            Initial positions of the atoms
        '''
        self.epsilon = epsilon
        self.sigma = sigma
        self.natom = natom
        self.x = np.array(x)
        if self.x.shape != (3*natom, ):
            raise TypeError('Initial positions does not have the correct size.')

    @property
    def dof(self):
        return 3*self.natom

    @property
    def hessian_is_approximate(self):
        return False

    def value(self):
        result = 0.0
        for iatom0 in xrange(self.natom):
            pos0 = self.x[3*iatom0:3*iatom0+3]
            for iatom1 in xrange(iatom0):
                pos1 = self.x[3*iatom1:3*iatom1+3]
                d = np.linalg.norm(pos0 - pos1)
                r = self.sigma/d
                result += self.epsilon*(r**12 - r**6)
        return result

    def gradient(self):
        result = np.zeros(self.x.shape)
        for iatom0 in xrange(self.natom):
            pos0 = self.x[3*iatom0:3*iatom0+3]
            g0 = result[3*iatom0:3*iatom0+3]
            for iatom1 in xrange(iatom0):
                pos1 = self.x[3*iatom1:3*iatom1+3]
                g1 = result[3*iatom1:3*iatom1+3]
                delta = pos0 - pos1
                dist = np.linalg.norm(delta)
                r = self.sigma/dist
                g = -6*self.epsilon*(2*r**12 - r**6)*(delta/dist**2)
                g0 += g
                g1 -= g
        return result

    def make_step(self, delta_x):
        self.old_x = self.x.copy()
        self.x += delta_x

    def step_back(self):
        self.x[:] = self.old_x

    def reset(self, x):
        self.old_x = self.x.copy()
        self.x[:] = x


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


def test_gradient():
    fn = NobleAtoms(1.0, 1.0, 10, np.zeros(30, float))
    with numpy_random_seed(2):
        xs = [np.random.normal(0, 7, 30) for ix in xrange(100)]
        fn.test_gradient(xs, nrep=4)
