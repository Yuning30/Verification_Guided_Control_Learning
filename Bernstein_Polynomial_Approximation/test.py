# from controller_approximation_lib import get_param
import numpy as np
import time
import os

def get_param():
    hidden_size = 2
    all_param = []
    init_param = np.random.rand(4*hidden_size+1).tolist()
    all_param += init_param
    all_param += [0, 1]
    all_param = [2, 1, 1, hidden_size] + all_param
    # print(all_param)
    return np.array(all_param) 

def gradient(cmd, parameter, goalset, step):
    gradient = []
    delta = np.random.uniform(low=-0.01, high=0.01, size=(parameter.size))
    index = np.random.randint(low=0, high=len(parameter))
    pert = np.zeros(len(parameter),)
    pert[index] += delta[index]
    
    start = time.time()
    # x = subprocess.run([cmd, str(step)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    os.system(cmd)

def get_real_time_param():
	global param
	return param

# if __name__ == '__main__':

param = get_param()
print(param)
gradient('./nn_os_relu_tanh 3', param, set([-0.05, -0.05, 0.05, 0.05]), 3)
param[5] = -10
param[6] = -10
param[7] = -10
gradient('./nn_os_relu_tanh 3', param, set([-0.05, -0.05, 0.05, 0.05]), 3)