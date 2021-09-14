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
    
    def _setup_simple_EKF_prediction(self):
        ## Initialize DAE integrator ##
        self.sim_p =sim_p= self.model.sv.sym_struct([
            entry('_u', struct=self.model._u),
            entry('_p', struct=self.model._p),
            entry('_tvp', struct=self.model._tvp),
            entry('_w', struct=self.model._w)])
        dae['x']=self.model._x
        dae['z']=self.model._z
        dae['p']=sim_p
        dae['ode']=self.model._alg
        dae['alg']=self.model._rhs
        ### Initialize ##
        self.sim_p=self.sim_p(0)
        #####
        self.x_estimated=vertcat(self.model._x,self.split_p(self.model._p)[0])
        #####
        self.get_estimated_states=Function('',[self.model._x,self.model._p],[self.x_estimated])
        #####
        self.rhs_estimated=get_estimated_states(self.rhs,self.model._p*0)
        #####  
        self.jacobian_of_rhs_for_estimation_x_est=Function('Jacobian_of_rhs_to_x_p_est',[self.x,self.z,simp],[jacobian(self.rhs_estimated,self.x_estimated)])
        #####
        self.jacobian_of_rhs_for_estimation_w_p  =Function('jacobian_of_rhs_for_estimation_w_p',[self.x,self.z,simp],[jacobian(self.rhs_estimated,vertcat(self.model._w,self.model._p))])
        #####
        self.jacobian_of_h_for_estimation = Function('Jacobian_of_h_to_x_p_est',[self.x,self.z,simp],[jacobian(self.model._y_expression,self.x_estimated)])
        ####
        self.integrator=integrator('F', 'idas', dae,{'tf':self.dt})
        return self.simple_EKF_prediction
    
    def _setup_simple_EKF_correction(self):
        return self.simple_EKF_correction
    
    def make_simple_EKF_prediction(self,simp):
        " Prediction "
        #### Calculate the Jacobian  ###
        jacobian_x_p=self.jacobian_of_rhs_for_estimation_x_est(self._x_num,self._z_num,simp)
        ####
        A=slin.expm(jacobian_x_p*self.dt) ##
        self.P_pre=((A@self._P_num)@A.T)+QP ## Prediction of the 
        self._x_num=integrator(x0=self._x_num ,z0=self._z_num, p = simp)['xf']
        ####
        
        
        
    def make_simple_EKF_correction(self):     
        " Correction "
        C=self.jacobian_of_h_for_estimation(self._x_num,self._z_num,self.simp) # Jacobian of the measurement function regarding to the predicted states
        S=(C@self._P_num@C.T)+R 
        K=(self._P_num@C.T)@slin.inv(S)#Calculation of the Kalman Gain 
        ###
        ## Correction of the P-Matrix
        self._x_num=self._x_num+K@(self._y_num-self.h(self._x_num,self._z_num,simp))
        ## Correction of the P-Matrix
        I=DM.eye(self._P_num.shape[0])# 
        self._P_num=(I-(K@H))@P_pre@(I-(K@H)).T+K@R@K.T#    
        y_post=h(self._x_num,z_pre,self.simp)# Measurement after the correction
         
