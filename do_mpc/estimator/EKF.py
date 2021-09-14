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
from do_mpc.estimator.estimator import Estimator 

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
        
    
