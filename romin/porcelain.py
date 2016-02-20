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
"""User-friendly SciPy-like API to the minimizer

Obviously, there are more advanced ways to interact with the minimizer that are not
possible with this simple API.
"""


from romin.hessian_models import SR1HessianModel
from romin.ntr import minimize_objective_ntr
from romin.objective import UserObjective, HessianModelWrapper


__all__ = ['minimizer_ntr']


def minimizer_ntr(x0, value, gradient, dot_hessian=None, grms_threshold=1e-7, maxiter=128,
        verbose=0):
    """Minimize a function with a (quasi) newton optimizer

    Parameters
    ----------
    x0 : np.ndarray, shape=(n,)
         The initial guess of the minimizer.
    value : function
            A function that computes the value of the objective for a given argument
            x.
    gradient : function
               A function that computes the gradient of the objective for a given
               argument x.
    dot_hessian : function (optional)
                  A function that computes the dot product of the Hessian of the
                  objective with a test vector y, for a given argument x:
                  dot_hessian(x, y). When not provided, a quasi-newton optimization is
                  carried out.
    grms_threshold : float
                     A convergence threshold for the root-mean-square value of the
                     gradient.
    maxiter : int
              The maximum number of iterations. When set to None, the number of iterations
              is unlimited.
    verbose : int
              Verbosity level. 0=silent, 1=normal, 2=chatterbox

    Returns
    -------
    x : np.ndarray, shape=(n,)
        The minimizer of the objective

    Raises
    ------
    ConvergenceFailure
        when the maximum number of iterations is reached before reaching convergence.
    """
    objective = UserObjective(x0, value, gradient, None, dot_hessian)
    if dot_hessian is None:
        mh = SR1HessianModel()
        wrapper = HessianModelWrapper(objective, mh)
        minimize_objective_ntr(wrapper, grms_threshold, maxiter, verbose)
    else:
        minimize_objective_ntr(objective, grms_threshold, maxiter, verbose)
    return objective.x
