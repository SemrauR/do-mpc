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

        self.data_fields = ["prediction_type", "correction_type", "constraint_handling_type", "dt", "tvp_fun"]
        self.prediction_type = "simple" # simple | 
        self.correction_type = "simple" # simple | 
        self.constraint_handling_type = "none" # none | simple | QP | NLP
        self.dt = 1
        
        # The full set of parameters is split in estimated and set parameters
        # To extract the estimated parameters multiply the full set by the permutation matrix
        # example : p_est = self._p_perm_p_est @ p_full
        # alternatively use the functions self._split_p or self._merge_p
        _p = model._p
        sym_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _p, model.sv, {"_p_est": [p_i for p_i in _p.keys() if p_i in p_est_list],
                           "_p_set": [p_i for p_i in _p.keys() if p_i not in p_est_list]}, "_p")
        self._p_est = sym_structs["_p_est"]
        self._p_set = sym_structs["_p_set"]
        self._split_p = split_fun
        self._merge_p = merge_fun
        self._p_perm_p_est = perm_mat_dict["_p_est"]
        self._p_perm_p_set = perm_mat_dict["_p_set"]

        # The full set of states is split in noisy (process noise w is added) and clean states
        # To extract the noisy states multiply the full set by the permutation matrix
        # example : x_noisy = self._s_perm_p_noisy @ x_full
        # alternatively use the functions self._split_x or self._merge_x
        _x = model._x
        sym_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _x, model.sv, {"_x_noisy": [p_i for p_i in _x.keys() if p_i in model._w.keys()],
                           "_x_clean": [p_i for p_i in _x.keys() if p_i not in model._w.keys()]}, "_x")
        self._x_noisy = sym_structs["_x_noisy"]
        self._x_clean = sym_structs["_x_clean"]
        self._split_x = split_fun
        self._merge_x = merge_fun
        self._x_perm_x_noisy = perm_mat_dict["_x_noisy"]
        self._x_perm_x_clean = perm_mat_dict["_x_clean"]
        
        # The full set of measurements is split in noisy (measurement noise v is added) and clean measurements
        # To extract the noisy states multiply the full set by the permutation matrix
        # example : x_noisy = self._s_perm_p_noisy @ x_full
        # alternatively use the functions self._split_x or self._merge_x        _y = model._y
        sym_structs, perm_mat_dict, merge_fun, split_fun = struct_split(
            _y, model.sv, {"_y_noisy": [p_i for p_i in _y.keys() if p_i in model._v.keys()],
                           "_y_clean": [p_i for p_i in _y.keys() if p_i not in model._v.keys()]}, "_y")
        self._y_noisy = sym_structs["_y_noisy"]
        self._y_clean = sym_structs["_y_clean"]
        self._split_y = split_fun
        self._merge_y = merge_fun
        self._y_perm_y_noisy = perm_mat_dict["_y_noisy"]
        self._y_perm_y_clean = perm_mat_dict["_y_clean"]
        
        # The vectors can be appropriately "structured" by passing them as an argument to the corresponding structure
        # i.e.: model._p(p) -> Returns a structure of parameters that can be indexed by parameter name
        # i.e.: self._p_est(p_est) -> Returns a structure of estimated parameters that can be indexed by parameter name
        
        # Initialize structure to hold the extended state (state variables, algebraict states and estimated parameters)
        self._x_extended = sv.sym_struct([entry("_x", struct = _x),
                                          entry("_z", struct = _z),
                                          entry("_p_est", struct = self._p_est)])
        self._x_extended_short = sv.sym_struct([entry("_x", struct = _x),
                                                entry("_p_est", struct = self._p_est)])
        
        self._u_extended = sv.sym_struct([entry("_u", struct = self.model._u),
                                          entry("_p_set", struct = self._p_set),
                                          entry("_tvp", struct = self._tvp_set)])
        
        # R and Q are the covariance matrices for the white noise gaussian processes of vectors v and [w; p_est]
        # The error covariance matrix P is 
        self._R = model.sv.struct([entry("v", shapestruct = (model._v, model._v))])
        self._Q = model.sv.struct([entry("Q", shapestruct = (self._x_extended, ))])
        self._P = model.sv.struct([entry("P", shapestruct = (self._x_extended, self._x_extended))])
        self._P_short = model.sv.struct([entry("P", shapestruct = (self._x_extended_short, self._x_extended_short))])
        
        
    def make_step(self, y0):
        """Main method during runtime. Pass the most recent measurement and
        retrieve the estimated state."""
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        ########## Get Values ##########
        # Get u,p,tvp merged into the simp
        # Get x,z, and p merged into x_est and p
        ################################
        self.make_prediction()
        self.make_correction()
        self.make_constraint_handling()
        
    
    def _setup_prediction_function(self):
        if self.prediction_type='simple':
            self.make_prediction = self._setup_simple_EKF_prediction()
        return 
    
    
    def _setup_correction_function(self):
        if 
            self.make_correction = self._setup_simple_EKF_correction()
        return
    
    
    def _setup_constraint_handling_function(self):
        self.make_constraint_handling =
        return
    
    
    def setup():
        self._setup_nominal_values()
        self._setup_prediction_function()
        self._setup_correction_function()
        self._setup_constraint_handling_function()
        return
    
    def _setup_nominal_values(self):
        self.sim_p = sim_p= self.model.sv.sym_struct([
            entry('_u', struct=self.model._u),
            entry('_p', struct=self.model._p),
            entry('_tvp', struct=self.model._tvp),
            entry('_w', struct=self.model._w)])
        self.sim_p_num =self.sim_p(0)
        #### INIT DAE ###
        self.dae['x']=self.model._x
        self.dae['z']=self.model._z
        self.dae['p']=sim_p
        self.dae['ode']=self.model._alg
        self.dae['alg']=self.model._rhs
        ####
        self.x_estimated=vertcat(self.model._x,self.split_p(self.model._p)[0])
        #####
        self.get_estimated_states=Function('',[self.model._x,self.model._p],[self.x_estimated])
        #####
        self.rhs_estimated=get_estimated_states(self.rhs,self.model._p*0)
        #####  
        
        p = self._merge_p(p_est = self._p_est, p_set = self._u_extended["_p_set"])
        rhs_kwargs = {"_x": self.x, "_z": self._z, "_p" : p,
                      "_u": self._u_extended["_u"], "_tvp": self._u_extended["_tvp"],
                      "_w": 0}
        rhs_dict = self.model._rhs_fun(**rhs_kwargs)
        rhs_jac_dict = self.model._rhs_jac_fun(**rhs_kwargs)
        alg_dict = self.model._alg_fun(**rhs_kwargs)
        alg_jac_dict = self.model._alg_jac_fun(**rhs_kwargs)
        meas_kwargs = {"_x": self.x, "_z": self._z, "_p" : p,
                      "_u": self._u_extended["_u"], "_tvp": self._u_extended["_tvp"],
                      "_v": 0}        
        meas_dict = self.model._meas_fun(**meas_kwargs)
        meas_jac_dict = self.model._meas_jac_fun(**meas_kwargs)
        
        # TODO continue here. The functions will probably throw an error due to output not existing (it's a dict )
        self.jac_rhs_ext_fun = Function("Jacobian_of_rhs_to_x_ext", [self._x_extended, self._u_extended], [rhs_jac], ["_x_extended", "_u_extended"], ["_jacobian"])
        self.jac_alg_ext_fun = Function("Jacobian_of_alg_to_x_ext", [self._x_extended, self._u_extended], [alg_jac], ["_x_extended", "_u_extended"], ["_jacobian"])
        self.jac_meas_ext_fun = Function("Jacobian_of_meas_to_x_ext", [self._x_extended, self._u_extended], [meas_jac], ["_x_extended", "_u_extended"], ["_jacobian"])        
        
        rhs_z = -mldivide(alg_jac_dict["_z"], alg_jac_dict["_x"]@rhs_dict["_rhs"])
        rhs_z_jac = jacobian(rhs_z, self._x_extended)
        rhs_p_est_jac = zeros(self._p_est.shape[0],self._x_extended.shape[0])
        #A_ext = vertcat() 
        #C_ext = 
        
        self.jac_rhs_=Function('Jacobian_of_rhs_to_x_p_est',[self.x,self.z,simp],[jacobian(self.rhs_estimated,self.x_estimated)])
        #####
        self.jacobian_of_rhs_for_estimation_w_p  =Function('jacobian_of_rhs_for_estimation_w_p',[self.x,self.z,simp],[jacobian(self.rhs_estimated,vertcat(self.model._w,self.model._p))])
        #####
        self.jacobian_of_h_for_estimation = Function('Jacobian_of_h_to_x_p_est',[self.x,self.z,sim_p],[jacobian(self.model._y_expression,self.x_estimated)])
        ####
        
        self.h# Measurement Function
        
    def _setup_simple_EKF_prediction(self):
        #### Integrator ####
        self.integrator=integrator('F', 'idas', dae,{'tf':self.dt})
        return self.simple_EKF_prediction
    
    def _setup_simple_EKF_correction(self):
        return self.simple_EKF_correction
    
    def make_simple_EKF_prediction(self,simp):
        " Prediction "
        #### Calculate the Jacobian  ###
        jacobian_x_p=self.jacobian_of_rhs_for_estimation_x_est(self._x_num,self._z_num,simp)
        jacobian_f_p_w=jacobian_of_rhs_for_estimation_w_p(self._x_num,self._z_num,simp)
        #### From cont to discrete   ####
        F=slin.expm(jacobian_x_p*self.dt) ##
        A=jacobian_x_p
        Q_cont=jacobian_f_p_w@self._num_Q@jacobian_f_p_w.T
        G=slin.expm(vertcat(horzcat(-A.T,A*0),horzcat(Q_cont,A))*self.dt)
        QP=G[0:Q_cont.shape[0],Q_cont.shape[0]:]
        ####
        self.P_pre=((F@self._P_num)@F.T)+QP ## Prediction of the 
        self._x_num=integrator(x0=self._x_num ,z0=self._z_num, p = simp)['xf']
        ####    ####
        
    def make_simple_EKF_correction(self):     
        " Correction "
        C=self.jacobian_of_h_for_estimation(self._x_num,self._z_num,self.simp) # Jacobian of the measurement function regarding to the predicted states
        S=(C@self._P_num@C.T)+R 
        K=(self._P_num@C.T)@slin.inv(S)#Calculation of the Kalman Gain 
        ###
        ## Correction of the P-Matrix
        self._x_num=self._x_num+K@(self._y_num-self.h(self._x_num,self._z_num,simp))
        ## Correction of the P-Matrix
        I=DM.eye(self._P_num.shape[0])                 # 
        self._P_num=(I-(K@H))@P_pre@(I-(K@H)).T+K@R@K.T#    
        y_post=h(self._x_num,z_pre,self.simp)          # Measurement after the correction
         
    def make_none_constraint_handling(self):
        " Not Doing anythig " 
        
        
        
        