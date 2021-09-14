# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:10:36 2021

@author: smfitama
"""
from casadi import *
import do_mpc 
from do_mpc.model import Model
from do_mpc.estimator import EKF

M = Model("discrete", symvar_type = "MX")

x = M.set_variable("_x", "x", shape = (3,1))
u = M.set_variable("_u", "u", shape = (2,1))
A = M.set_variable("_p", "A", shape = (3,3))
B = M.set_variable("_p", "B", shape = (3,2))
x_next = A@x + B@u
M.set_rhs("x", x_next, process_noise = True)
M.set_meas("y", x*1, meas_noise = True)

M.setup()

esti = EKF(M, ["A"])
