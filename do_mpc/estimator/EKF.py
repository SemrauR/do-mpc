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

def struct_split(parent_struct, sv, children_dict, parent_name):
    # Convenience function to construct substructures from a list of elements of a parent struct and the appropriate helper functions
    _p = parent_struct
    children_structs_dict = {}
    perm_mat_dict = {}
    for child, var_list in children_dict.items():
        v  = sv.sym_struct(
                [entry('default', shape=(0,0))]+
                [entry(v_i, shape=_p[v_i].shape) for p_i in _p.keys() if v_i in var_list]
            )
        perm_mat = DM(jacobian(v(vertcat(*[reshape(_p[v_i],-1,1) for v_i in _p.keys() if v_i in var_list])),
                                         _p).sparsity())
        children_structs_dict[child] = v
        perm_mat_dict[child] = perm_mat
        
    split_p = Function("split" + parent_name, 
                       [_p], 
                       [perm@_p for perm in perm_mat_dict.values()], 
                       [parent_name], 
                       children_dict.keys())
    merge_p = Function("merge" + parent_name, 
                       [v for v in children_structs_dict.values()], 
                       [sum([perm_mat_dict[child].T@children_structs_dict[child] for child in children_dict.keys()])], 
                       children_dict.keys(), 
                       [parent_name])        
    return children_structs_dict, perm_mat_dict, merge_fun, split_fun

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
<<<<<<< HEAD
        # Initialize structure to hold the optimial solution and initial guess:
        self._opt_x_num = None
        # Initialize structure to hold the parameters for the optimization problem:
        self._opt_p_num = None
        
        self._opt_R_struct = None
        self._opt_Q_struct = None
        self._opt_P_struct = None
=======
>>>>>>> 364922dbe5460f0b367c2718c2fdec5b694b40a4
        
        self.data_fields = ["prediction_type", "correction_type", "constraint_handling_type"]
        self.prediction_type = "simple" # simple | 
        self.correction_type = "simple" # simple | 
        self.constraint_handling_type = "none" # none | simple | QP | NLP
        
        # Create seperate structs for the estimated and the set parameters (the union of both are all parameters of the model.)
        _p = model._p
        # Construction of the permutation matrices 
        # The permutation matrices work as follows:
            # p_est = _p_perm_p_est @ p (extract the vector of estimated parameters from the full vector of parameters)
            # p_set = _p_perm_p_set @ p (extract the vector of set parameters from the full vector of parameters)
            # p = _p_perm_p_est.T @ p_est + _p_perm_p_set.T @ p_set (merge the vectors of estimated and set parameters into the full vector of parameters)

        _p_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _p, model.sv, {"_p_est": [p_i for p_i in _p.keys() if p_i in p_est_list],
                           "_p_set": [p_i for p_i in _p.keys() if p_i not in p_est_list]}, "_p")
        self._p_est = _p_struct["_p_est"]
        self._p_set = _p_struct["_p_set"]
        self._split_p = split_fun
        self._merge_p = merge_fun
        self._p_perm_p_est = perm_mat_dict["_p_est"]
        self._p_perm_p_set = perm_mat_dict["_p_set"]
        
        _x = model._x
        _x_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _x, model.sv, {"_x_noisy": [p_i for p_i in _x.keys() if p_i in model._w.keys()],
                           "_x_clean": [p_i for p_i in _x.keys() if p_i not in model._w.keys()]}, "_x")
        self._x_noisy = _p_struct["_x_noisy"]
        self._x_clean = _p_struct["_x_clean"]
        self._split_x = split_fun
        self._merge_x = merge_fun
        self._x_perm_x_noisy = perm_mat_dict["_x_noisy"]
        self._x_perm_x_clean = perm_mat_dict["_x_clean"]
        
        _y = model._y
        _y_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _y, model.sv, {"_y_noisy": [p_i for p_i in _y.keys() if p_i in model._v.keys()],
                           "_y_clean": [p_i for p_i in _y.keys() if p_i not in model._v.keys()]}, "_y")
        self._y_noisy = _p_struct["_y_noisy"]
        self._y_clean = _p_struct["_y_clean"]
        self._split_y = split_fun
        self._merge_y = merge_fun
        self._y_perm_y_noisy = perm_mat_dict["_y_noisy"]
        self._y_perm_y_clean = perm_mat_dict["_y_clean"]
        
        # The vectors can be appropriately "structured" by passing them as an argument to the corresponding structure
        # i.e.: model._p(p) -> Returns a structure of parameters that can be indexed by parameter name
        # i.e.: self._p_est(p_est) -> Returns a structure of estimated parameters that can be indexed by parameter name
        

        
        # Initialize structure to hold the current estimate of the state
        self._x_num = model._x(0)
        # Initialize structure to hold the current estimate of the full parameter set:
        self._p_num = model._p(0)
        
        self._R_num = model.sv.struct([entry("R", shapestruct = (model._v, model._v))])(0)
        self._Q_num = model.sv.struct([entry("Q", shapestruct = (model._w, model._w))])(0)
        self._P_num = model.sv.struct([entry("P", shapestruct = (model._w,self._p_est, model._w))])(0)
        
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
         
