#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import copy
import warnings
import time
import do_mpc.optimizer
import do_mpc.data
from do_mpc.estimator import Estimator 

class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    .. warning::
        Not currently implemented.
    """
    def __init__(self, model, p_est_list):
        super().__init__(model)
        # Flags are checked when calling .setup.
        self.flags = {
            'setup': False,
        }
        
        # Initialize structure to hold the optimial solution and initial guess:
        self._opt_x_num = None
        # Initialize structure to hold the parameters for the optimization problem:
        self._opt_p_num = None
        
        self._opt_R_struct = None
        self._opt_Q_struct = None
        self._opt_P_struct = None
        
        self.data_fields = ["prediction_type", "correction_type", "constraint_handling_type"]
        self.prediction_type = "simple" # simple | 
        self.correction_type = "simple" # simple | 
        self.constraint_handling_type = "none" # none | simple | QP | NLP
        
        # Create seperate structs for the estimated and the set parameters (the union of both are all parameters of the model.)
        _p = model._p
        self._p_est  = self.model.sv.sym_struct(
            [entry('default', shape=(0,0))]+
            [entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i in p_est_list]
        )
        self._p_set  = self.model.sv.sym_struct(
            [entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i not in p_est_list]
        )
        # Construction of the permutation matrices 
        # The permutation matrices work as follows:
            # p_est = _p_perm_p_est @ p (extract the vector of estimated parameters from the full vector of parameters)
            # p_set = _p_perm_p_set @ p (extract the vector of set parameters from the full vector of parameters)
            # p = _p_perm_p_est.T @ p_est + _p_perm_p_set.T @ p_set (merge the vectors of estimated and set parameters into the full vector of parameters)

        self._p_perm_p_est = DM(jacobian(self._p_est(vertcat(*[reshape(_p[p_i],-1,1) for p_i in _p.keys() if p_i in p_est_list])),
                                         _p).sparsity())
        self._p_perm_p_set = DM(jacobian(self._p_set(vertcat(*[reshape(_p[p_i],-1,1) for p_i in _p.keys() if p_i not in p_est_list])),
                                         _p).sparsity())
        
        # The vectors can be appropriately "structured" by passing them as an argument to the corresponding structure
        # i.e.: model._p(p) -> Returns a structure of parameters that can be indexed by parameter name
        # i.e.: self._p_est(p_est) -> Returns a structure of estimated parameters that can be indexed by parameter name
        
        # Helper functions to convert one set of parameters into the other
        self._split_p = Function("split_p", [_p], [self._p_perm_p_est@_p, self._p_perm_p_set@_p], ["_p"], ["_p_est", "_p_set"])
        self._merge_p = Function("merge_p", [self._p_est, self._p_set], [self._p_perm_p_est.T@self._p_est + self._p_perm_p_set.T@self._p_set], ["_p_est", "_p_set"], ["_p"])
        
        
    def make_step(self, y0):
        """Main method during runtime. Pass the most recent measurement and
        retrieve the estimated state."""
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        None
        
    
    def _setup_prediction_function(self):
        return 
    
    
    def _setup_correction_function(self):
        return
    
    
    def _setup_constraint_handling_function(self):
        return
    
    
    def setup():
        self._setup_prediction_function()
        self._setup_correction_function()
        self._setup_constraint_handling_function()
        
        return
        
    def make_simple_EKF_prediction(self):
        " Prediction "
        x_pre=integrator(x0=x0 , p = vertcat(u,p,tv_p))['xf']
        #### Calculate the Jacobian  ###
        jacobian_of_rhs_wrt_x=self.model._rhs_jac_fun(x0,)
        ####
        A=slin.expm(jacobian_x_p*self.dt) ##
        P_pre=((A@P_post)@A.T)+QP ## Prediction of the 
        
    def make_simple_EKF_correction(self):     
        " Correction "
        "H=del h(x,u,p,tv_p)/ del x,p_est"
        H= # Jacobian of the measurement function regarding to the predicted states
        S=(C@P_pre@C.T)+R 
        K=(P_pre@C.T)@slin.inv(S)#Calculation of the Kalman Gain 
        ###
        y_pre=h(x_pre,vertcat(u,p,tv_p))# Measurement after the prediction
        ## Correction of the P-Matrix
        x_post=x_pre+K@(y-yp)
        ## Correction of the P-Matrix
        I=DM.eye(P_pre.shape[0])# 
        P_post=(I-(K@H))@P_pre@(I-(K@H)).T+K@R@K.T#    
        y_post=h(x_post,vertcat(u,p,tv_p))# Measurement after the correction
        return 
