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


def test_multi_gauss_diis():
    for irep in xrange(100):
        with numpy_random_seed(irep):
            reference, xs = get_problem(5)
            converged_a, counter_a, amps_a = run_mult_gauss_diis(reference, xs, 200, False)
            converged_b, counter_b, amps_b = run_mult_gauss_diis(reference, xs, 200, True)
            assert counter_a >= counter_b
            assert converged_b
            if converged_a and converged_b:
                assert abs(amps_a - amps_b).max() < 1e-10


def get_problem(nsite):
    reference = []
    for isite in xrange(nsite):
        reference.append((isite, np.random.uniform(0.5, 0.8), np.random.randint(50, 150)))
    xs = np.concatenate([np.random.normal(x0, s0, size) for x0, s0, size in reference])
    return reference, xs


def run_mult_gauss_diis(reference, xs, maxiter, do_diis):
    # This test is an iterative refinement of a univariate multi-Gaussian distribution.
    # The widths and centers are kept fixed but the amplitudes are fitted to best match
    # the observed samples. In this test, the samples are randomly generated.
    
    basis = []
    for x0, s0, size in reference:
        f = np.exp(-0.5*((xs-x0)/s0)**2) / np.sqrt(2*np.pi) / s0
        basis.append(f)
    
    def update(amps):
        rho0 = sum(amp*f for amp, f in zip(amps, basis))
        dots = np.array([(f/rho0).sum() for f in basis])
        gradient = -dots + 1
        new_amps = dots*amps
        return gradient, new_amps
    
    def check(amps):
        return (amps > 0).all()

    amps0 = np.ones(len(reference), float)
    diis = DIIS(5)
    counter = 0
    while counter < maxiter:
        amps1 = update(amps0)[1]
        gradient1 = update(amps1)[0]
        counter += 1
        #print counter, 1, rms(gradient1), amps1
        if do_diis:
            diis.feed(amps1, gradient1)
            amps2 = diis.guess(downhill=True, validation=check)
        else:
            amps2 = None
        if amps2 is None:
            amps2 = amps1
            gradient2 = gradient1
        else:
            gradient2 = update(amps2)[0]
            counter += 1
            #print counter, 2, rms(gradient2), amps2
        if rms(gradient2) < 1e-15:
            return True, counter, amps2
        amps0 = amps2
    return False, counter, amps2
