from ray_tracer import *

pos_ini = [15.,0.,0.,]
vel_ini = [10.,10.,10.,]
k_test = [-1.,-1.,-1.,]
xh = 0.1  # Rs
kh = 1e-6  # 1/m
query_ini_k(pos_ini,k_test,vel_ini,xh,kh,mode='Slow')