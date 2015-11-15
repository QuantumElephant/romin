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
from horton import log


__all__ = ['minimize_ntr']


def minimize_ntr(objective, grms_threshold=1e-8, maxiter=128):
    converged = False
    counter = 0
    trust_radius = 1.0
    value = objective.value()
    gradient = objective.gradient()
    if log.do_high:
        log('Number of unknowns: %i' % objective.dof)
        log.blank()
    if objective.dof == 0:
        log('Since the number of unknowns is 0, there is nothing to optimize.')
        return 0

    trs = CGTrustRadiusSolver(objective, gradient, trust_radius, grms_threshold)
    while maxiter is None or counter < maxiter:
        # Status at beginning of iteration
        grms = rms(gradient)
        if log.do_high:
            log('Counter:      %17i' % counter)
            log('Value:        %17.10f' % value)
            log('Gradient rms: %17.10e / %12.5e' % (grms, grms_threshold))
        elif log.do_medium:
            log('%4i  %17.10e  %12.5e  %12.5e' % (counter, value, grms, trust_radius))

        # Check for convergence
        if grms < grms_threshold:
            log('CONVERGED!')
            converged = True
            break

        # If we got here, construct a new step with the TRS.
        if log.do_high:
            log.blank()
            log('   Trust radius: %.5e' % trust_radius)
        delta, estimated_change, estimated_g = trs.solve()

        if True:
            # Sanity checks on the results from the trs, cost time
            assert abs(estimated_change - (
                np.dot(delta, gradient)
                +0.5*np.dot(delta, objective.dot_hessian(delta))
            )) < 1e-7
            assert abs(estimated_g - gradient - objective.dot_hessian(delta)).max() < 1e-7

        # Compute value and gradient at new point
        objective.make_step(delta)
        new_value = objective.value()
        new_gradient = objective.gradient()
        if log.do_high:
            log('   Estimated objective change: %17.10e' % estimated_change)
            log('   Actual objective change:    %17.10e' % (new_value - value))
            log('   Estimated gradient RMS:  %17.10e' % rms(estimated_g))
            log('   Actual    gradient RMS:  %17.10e' % rms(new_gradient))

        # Check if step is acceptable
        acceptable = True
        if new_value >= value:
            acceptable = False
            if log.do_high:
                log('   The objective did not decrease!')
            if abs(estimated_change) < 1e-13*abs(value) and \
               rms(new_gradient) < rms(gradient):
                acceptable = True
                if log.do_high:
                    log('   However, the estmated value change is tiny and the gradient decreased.')

        if not acceptable:
            if log.do_high:
                log('   Restarting step with reduced trust radius.')
                log.blank()
            # really not good enough:
            # - do not accept step
            objective.step_back()
            # - increase step counter
            counter += 1
            # - decrease trust radius a lot
            trust_radius *= 0.5
            # If the objective has an approximate Hessian, i.e. dependent on previous
            # calculations, the cache of the trust-radius solver must be dropped.
            if objective.hessian_is_approximate:
                trs.drop_cache()
            trs.trust_radius = trust_radius
            continue

        if log.do_high:
            log('   Accepting step.')

        # 30% deviation on the estimated value change or gradient are
        # considered to be within the trust region. It is OK for the value
        # to be more than 30% below the estimated value.
        v_crit = (new_value - (value + estimated_change))/abs(estimated_change)
        g_crit = rms(estimated_g - new_gradient)/rms(new_gradient)
        if log.do_high:
            log('   Value Criterion:         %15.1f%%' % (v_crit*100))
            log('   Gradient Criterion:      %15.1f%%' % (g_crit*100))
        if v_crit > 0.3 or g_crit > 0.3:
            if log.do_high:
                log('   Poor extrapolation of value or gradient!')
                log('   Trust radius will be reduced.')
            trust_radius *= 0.7
            if g_crit > 3.0:
                if log.do_high:
                    log('   Gradient criterion is very high!')
                    log('   Extra reduction of trust radius.')
                trust_radius *= 0.5
        else:
            if log.do_high:
                log('   Good extrapolations!')
                log('   Trust radius will be slightly increased.')
            trust_radius *= 1.3
        if log.do_high:
            log.blank()

        # If v_crit is way off and the hessian is approximate, do a Hessian reset.
        if objective.hessian_is_approximate and abs(v_crit) > 10:
            if log.do_high:
                log('Resetting Hessian because of large value criterion.')
            objective.reset_hessian()

        # increase step counter
        counter += 1
        # accept step
        value = new_value
        gradient = new_gradient
        trs = CGTrustRadiusSolver(objective, gradient, trust_radius, grms_threshold)

    if not converged:
        raise RuntimeError('Convergence failed')

    return counter


def rms(arr):
    '''Computes the RMS value of the array elements'''
    return np.sqrt((arr**2).sum()/arr.size)


class CGTrustRadiusSolver(object):
    def __init__(self, objective, gradient, trust_radius, grms_threshold):
        self.objective = objective
        self.gradient = gradient
        self.trust_radius = trust_radius
        self.grms_threshold = grms_threshold
        self.cache = {}

    def drop_cache(self):
        self.cache = {}

    def dot_hessian(self, x):
        key = tuple(x)
        result = self.cache.get(key)
        if result is None:
            result = self.objective.dot_hessian(x)
            self.cache[key] = result.copy()
        return result

    def go_to_trust_radius(self, x0, direction):
        '''Move from a given point to the trust radius, following a given direction

           Parameters
           ----------
           x0 : np.ndarray, dtype=float
                Current solution relative to the center of the trust sphere.
           direction : np.ndarray, dtype=float
                       The down-hill direction to follow.

           Returns
           -------
           x1 : np.ndarray, dtype=float
                Point on the trust sphere relative to the center of the sphere.
        '''
        direction_sq = np.dot(direction, direction)
        radius_sq = np.dot(x0, x0)
        ratio = np.dot(x0, direction)/direction_sq
        # error = current (signed) distance squared from the sphree.
        error = (self.trust_radius**2 - radius_sq)/direction_sq
        # x0 should be in the sphere:
        assert error > 0
        # alpha = distance to the sphere in units of `direction`. Positive solution is
        # the one of interest.
        alpha = (ratio**2 + error)**0.5 - ratio
        return x0 + alpha*direction, alpha

    def solve(self):
        '''Find a solution on the trust sphere, using the conjugate gradient method

           This solver uses the origin of the sphere as the initial guess. It then updates
           the solution using the conjugate gradient method, where the maximum number of
           steps equals the dimensionality of the problem. There are three exit
           conditions:

           1) The maximum number of steps is reached
           2) The RMS of the residual gradient is below ``grms_threshold``
           3) As CG step goes outside the trust sphere

           In the last case, the intersection of trust sphere and the line between the two
           last solutions as the final solution

           Returns
           -------
           solution : np.ndarray, dtype=float
                      The solution of the CG solver, relative to the origin of the trust
                      sphere.
           change : float
                    The estimated decrease of the objective, using the (possibly
                    approximate) Hessian.
           gradient : np.ndarray, dtype=float
                      The estimated gradient at the solution. This is the approximate
                      gradient, derived from the (possibly approximate) Hessian and the
                      solution relative to the center of the sphere.
        '''
        # Initialization of the CG algorithm
        if log.do_high:
            log('   Iter        Radius  Residual RMS      E Change')
        solution = np.zeros(len(self.gradient))
        residual = -self.gradient
        direction = residual.copy()
        residual_sq = np.dot(residual, residual)
        residual_rms = (residual_sq/len(residual))**0.5
        radius_sq = 0.0
        status = 'imax'
        alpha = 0.0
        beta = 0.0
        change = 0.0

        # Main CG loop
        for irep in xrange(len(self.gradient)):
            # Check for convergence
            if residual_rms < self.grms_threshold:
                status = 'conv'
                break

            # Compute the estimated change in objective and report
            new_change = 0.5*np.dot(self.gradient - residual, solution)
            if log.do_high:
                log('   %4i  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e' % (
                    irep, radius_sq**0.5, residual_rms, new_change,
                    alpha, beta
                ))

            # Do not allow an increase in the estimated change of objective. This would
            # correspond to a bug.
            if (new_change - change) > 1e-7*abs(change):
                raise AssertionError('The estimated change in objective increased '
                                     'from %.5e to %.5e.' % (change, new_change))
            change = new_change
            dot_direction = self.dot_hessian(direction)
            alpha = np.dot(residual, direction)/np.dot(direction, dot_direction)
            if alpha > 0:
                # Regular CG
                new_solution = solution + alpha*direction
                new_residual = residual - alpha*dot_direction
                new_residual_sq = np.dot(new_residual, new_residual)
                new_residual_rms = (new_residual_sq/len(residual))**0.5
                beta = new_residual_sq/residual_sq
                new_direction = new_residual + beta*direction
            else:
                beta = 0.0
                # Follow downhil along negative curvature until hitting
                # Trust radius. Then break.
                solution, x = self.go_to_trust_radius(solution, direction)
                assert x >= 0
                status = 'curv'
                break
            # Check if we are still within the trust region
            radius_sq = np.dot(new_solution, new_solution)
            if radius_sq > self.trust_radius**2:
                # Interpolate between new and old solution to end up at trust
                # radius. Then break
                solution, x = self.go_to_trust_radius(solution, new_solution - solution)
                assert x >= 0
                assert x <= 1
                status = 'trad'
                break
            # Prepare for next iteration
            solution = new_solution
            residual = new_residual
            residual_sq = new_residual_sq
            residual_rms = new_residual_rms
            direction = new_direction

        # Final checks and screen output
        radius_sq = np.dot(solution, solution)
        residual = -self.gradient-self.dot_hessian(solution)
        new_change = 0.5*np.dot(self.gradient - residual, solution)
        if log.do_high:
            residual_sq = np.dot(residual, residual)
            residual_rms = (residual_sq/len(residual))**0.5
            log('   %4s  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e' % (
                status, radius_sq**0.5, residual_rms, new_change,
                alpha, beta
            ))

        # Do not allow an increase in the estimated change of the objective. That would
        # correspond to a bug.
        if (new_change - change) > 1e-7*abs(change):
            raise AssertionError('The estimated change in objective increased '
                                 'from %.15e to %.15e.' % (change, new_change))
        change = new_change

        # Done
        return solution, change, -residual
