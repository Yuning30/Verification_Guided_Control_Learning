import error_analysis as ea
import sympy as sp
import ast
from network_parser import nn_controller_details
from neuralnetwork import NN
import numpy as np
import torch
from model import Actor
import time
import subprocess
import os
# from test import get_real_time_param
import warnings
import sinkhorn_pointcloud as spc
warnings.filterwarnings("ignore")


def get_param():
    hidden_size = 2
    all_param = []
    # OS
    init_param = np.random.rand(4*hidden_size+1).tolist()
    
    # init_param = np.random.uniform(low=-0.003, high=0.003,size=(4*hidden_size+1,)).tolist()
    all_param += init_param
    # Oscillator
    all_param += [0, 1]
    # ACC
    # all_param += [0, 50]
    all_param = [2, 1, 1, hidden_size] + all_param
    all_param[6] = 0
    all_param[9] = 0
    all_param[12] = 0
    # print(all_param)
    return np.array(all_param) 


def poly_approx_controller(
    d_str,
    box_str,
    output_index,
    activation,
    nerual_network
):
    param = np.load('param.npy')
    # print(param)
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    # print(param)
    nn = NN(param, activation, reuse=True)
    x = sp.symbols('x:' + str(nn.num_of_inputs))
    b, _, _ = ea.nn_poly_approx_bernstein(nn.controller, x, d, box, output_i)
    return ea.p2c(b)


def poly_approx_error(
    d_str,
    box_str,
    output_index,
    activation,
    nerual_network
):  
    param = np.load('param.npy')
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    nn = NN(param, activation, reuse=True)
    error_bound = ea.bernstein_error_partition_cuda(
        nn,
        nn.controller,
        d,
        box,
        output_i,
        activation,
        nerual_network
    )
    return ea.p2c(error_bound)

class set:
    def __init__(self, x):
        assert x[2] > x[0]
        assert x[3] > x[1]
        self.x_inf = x[0]
        self.y_inf = x[1]
        self.x_sup = x[2]
        self.y_sup = x[3]
        self.area = (self.x_sup - self.x_inf)*(self.y_sup - self.y_inf)

def intersect(set1, set2):
    if set1.x_sup <= set2.x_inf or set1.x_inf >= set2.x_sup or set1.y_sup <= set2.y_inf or set1.y_inf >= set2.y_sup:
        return False
    else:
        return True

def W_distance(reachset, targetset):
    N = 100
    x0 = np.random.uniform(reachset.x_inf, reachset.x_sup, size=(N,))
    x1 = np.random.uniform(reachset.y_inf, reachset.y_sup, size=(N,))
    x = np.vstack((x0, x1))

    y0 = np.random.uniform(targetset.x_inf, targetset.x_sup, size=(N,))
    y1 = np.random.uniform(targetset.y_inf, targetset.y_sup, size=(N,))
    y = np.vstack((y0, y1))

    X = torch.FloatTensor(x.T)
    Y = torch.FloatTensor(y.T)
    # OS
    epsilon = 0.01
    niter = 100

    # ACC
    epsilon = 100
    niter = 200
    l1 = spc.sinkhorn_loss(X,Y,epsilon,N,niter)
    # l2 = spc.sinkhorn_normalized(X,Y,epsilon,N,niter)
    if intersect(reachset, targetset): 
        print('ohh, intersect and W distance is: ', l1.item())
    else:
        print('not intersect and W distance is： ', l1.item())   
    return l1.item()


def metric(reachset, targetset):
    # the width and height of 2 rectangles
    r_width = reachset.x_sup - reachset.x_inf
    r_height = reachset.y_sup - reachset.y_inf
    g_width = targetset.x_sup - targetset.x_inf 
    g_height = targetset.y_sup - targetset.y_inf

    # the center position of two rectangles
    r_x = (reachset.x_inf + reachset.x_sup) / 2
    r_y = (reachset.y_inf + reachset.y_sup) / 2
    g_x = (targetset.x_inf + targetset.x_sup) / 2
    g_y = (targetset.y_inf + targetset.y_sup) / 2
    
    if intersect(reachset, targetset):
        # to compute the intersection area
        overlapX = r_width + g_width - (max(reachset.x_sup, targetset.x_sup)-min(reachset.x_inf, targetset.x_inf))
        overlapY = r_height + g_height - (max(reachset.y_sup, targetset.y_sup)-min(reachset.y_inf, targetset.y_inf))
        assert overlapX >= 0
        assert overlapY >= 0
        area = overlapX*overlapY
        print("Ohoo, intersected and the area is: ", area)
        return -area
    else:
        # to compute the minimal distance between two sets
        # the distance of two centers on x, y
        Dx = abs(r_x - g_x)
        Dy = abs(r_y - g_y)

        if Dx < (r_width+g_width) / 2 and Dy >= (r_height+g_height) / 2:
            min_dist = Dy - (r_height+g_height) / 2
        elif Dx >= (r_width+g_width) / 2 and Dy < (r_height+g_height) / 2:
            min_dist = Dx - (r_width+g_width) / 2
        elif Dx >= (r_width+g_width) / 2 and Dy >= (r_height+g_height) / 2:
            delta_x = Dx - (r_width+g_width) / 2
            delta_y = Dy - (r_height+g_height) / 2
            min_dist = np.sqrt(delta_x**2+delta_y**2)
        print("Not intersect and min distance is: ", min_dist)
        return min_dist

def gradient(cmd, parameter, goalset, step, measure,unsafecase=False):
    unsafe_set = set([-0.3, 0.2, -0.25, 0.35])
    gradient = []
    if unsafecase:
        # OS
        delta = np.random.uniform(low=-0.2, high=0.2, size=(parameter.size))
        # delta = np.random.uniform(low=-0.5, high=0.5, size=(parameter.size))
    else:
        # OS
        delta = np.random.uniform(low=-0.05, high=0.05, size=(parameter.size))
        # delta = np.random.uniform(low=-0.5, high=0.5, size=(parameter.size))
    index = np.random.randint(low=0, high=len(parameter))
    while(index not in [4, 5, 7, 8, 10, 11]):
        index = np.random.randint(low=0, high=len(parameter))
    pert = np.zeros(len(parameter),)
    pert[index] += delta[index]
    
    start = time.time()
    np.save('param.npy', parameter+pert)
    x = subprocess.run([cmd, str(step)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print('here:', x, len(x))
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    # if x[0] >= 120 and x[1] >= 25:
    #     assert False
    reachset1 = set(x)
    m1 = measure(reachset1, goalset)

    m_unsafe = measure(reachset1, unsafe_set)

    # os.system('./nn_os_relu_tanh 1')

    np.save('param.npy', parameter-pert)
    x = subprocess.run([cmd, str(step)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    # if x[0] >= 120 and x[1] >= 25:
    #     assert False
    reachset2 = set(x)
    m2 = measure(reachset2, goalset)

    # os.system('./nn_os_relu_tanh 1')
    
    gradient = np.zeros(len(parameter), )
    gradient[index] = 0.5*(m1-m2)/delta[index] 
    # gradient[index] -= 0.0005*(reachset1.area-reachset2.area)/delta[index]
    
    assert index in [4,5,7,8,10,11]
    return gradient, m1, m2, intersect(reachset1, goalset), intersect(reachset2, goalset), m_unsafe

if __name__ == '__main__':

    now = time.time()


#Oscillator    
    ### heuristic metrics based ###
    # # initilize the controller parameters
    # param = get_param()
    # # param = np.load('param.npy')
    # param_history = []
    # # gradient descent 
    # for i in range(500):
    #     print('here begins the ' + str(i) + 'th updates:')
    #     param_history.append(param)
    #     # goal_only 
    #     gra, _, g1, intergoal = gradient('./nn_os_relu_tanh', param, set([-0.05, -0.05, 0.05, 0.05]), 15, measure=metric)

    #     param -= gra
    #     # if i > 60:
    # # also consider the unsafe set
    #     safe, _, s1, intersafe = gradient('./nn_os_relu_tanh', param, set([-0.3, 0.2, -0.25, 0.35]), 6, measure=metric,unsafecase=True)
    #     # print(safe)

    #     param +=  0.05 * safe
    #     if intergoal and not intersafe:
    #         break

    ### Wasserstain Distance
    # # initilize the controller parameters
    param = get_param()

    # param = np.load('param.npy')
    param_history = []
    # gradient descent
    goal_w = []
    safe_w = []
    for i in range(2):
        print('')
        print('')
        print('here begins the ' + str(i) + 'th updates:')
        # param_history.append(param)
        # goal_only 
        goal_gra, goal1, goal2, goal_inter, goal_inter_pri, safe1 = gradient('./nn_os_relu_tanh', param, set([-0.05, -0.05, 0.05, 0.05]), 15, measure=W_distance)
        goal_w.append((goal1+goal2)/2)
        # print(goal_gra)
        goal_gra = np.clip(goal_gra, -1.5, 1.5)
        param -= goal_gra

        safe_gra, _, _, safe_inter, safe_inter_pri,  _= gradient('./nn_os_relu_tanh', param, set([-0.3, 0.2, -0.25, 0.35]), 6, unsafecase=True, measure=W_distance)
        safe_w.append(safe1)
        if goal_inter:
            safe_gra = np.clip(safe_gra, -1.5, 1.5)
            param +=  safe_gra

        if goal_inter and not safe_inter:
            np.save('goal_w_'+str(i)+'_'+str(int(time.time() - now))+'.npy', np.array(goal_w))
            np.save('safe_w_'+str(i)+'_'+str(int(time.time() - now))+'.npy', np.array(safe_w))
            print('The running time is: ', time.time() - now)
            break

        if goal_inter_pri and not safe_inter_pri:
            np.save('goal_w_'+str(i)+'_'+str(int(time.time() - now))+'.npy', np.array(goal_w))
            np.save('safe_w_'+str(i)+'_'+str(int(time.time() - now))+'.npy', np.array(safe_w))
            print('The running time is: ', time.time() - now)
            break



# # ACC
#     ### Heuristic metrics
#     # # initilize the controller parameters
#     param = get_param()

#     param = np.load('param_acc.npy')
#     param_history = []
#     # gradient descent 
#     for i in range(500):
#         print('here begins the ' + str(i) + 'th updates:')
#         # param_history.append(param)
#         # goal_only 
#         # gra, _, _ = gradient('./nn_acc_relu_tanh', param, set([145, 39, 155, 41]), 6, measure=metric)
#         # print(gra)
#         # gra = np.clip(gra, -200, 200)
#         # param -= 0.01 * gra
#         # assert False

#     #     if i > 60:
#     # also consider the unsafe set
#         safe_dis_gra, _, _, inter_dis = gradient('./nn_acc_relu_tanh', param, set([100, 20, 120, 60]), 6, measure=metric,unsafecase=True)
#         print('safe_distance_gradient: ', safe_dis_gra)
#         # safe_v_gra, _, _ = gradient('./nn_acc_relu_tanh', param, set([120, 0, 180, 25]), 6, measure=metric,unsafecase=True)
#         # print(safe_v_gra)
#         safe_dis_gra = np.clip(safe_dis_gra, -20, 20)
#         if inter_dis:
#             param += 0.01 * safe_dis_gra

        # if not inter_dis:
        #     safe_v_gra, _, _, inter_v = gradient('./nn_acc_relu_tanh', param, set([100, 0, 150, 25]), 6, measure=metric,unsafecase=True)
        #     print('safe_velocity_gradient: ', safe_v_gra)
        #     safe_v_gra = np.clip(safe_v_gra, -5, 5)
        #     param += 0.1 * safe_v_gra 

        # if not inter_dis and not inter_v:
        #     assert False                
