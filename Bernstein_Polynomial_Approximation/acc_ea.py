import sys 
sys.path.append("..") 
import os
import argparse
import math
import h5py
import tensorflow as tf
# import tf_util as U
import numpy as np
import sympy as sp

from scipy.special import comb
from numpy import linalg as LA
from scipy.optimize import linprog
# from polyval import polyval
import subprocess
from itertools import product
from accenv import ACC
from layers import SinkhornDistance
import torch
import time


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

def W_distance(reachset, targetset, printC = False):
    N = 100
    print(reachset.x_inf, reachset.x_sup, reachset.y_inf, reachset.y_sup)
    x0 = np.random.uniform(reachset.x_inf, reachset.x_sup, size=(N,))
    x1 = np.random.uniform(reachset.y_inf, reachset.y_sup, size=(N,))
    x = np.vstack((x0, x1)).T

    y0 = np.random.uniform(targetset.x_inf, targetset.x_sup, size=(N,))
    y1 = np.random.uniform(targetset.y_inf, targetset.y_sup, size=(N,))
    y = np.vstack((y0, y1)).T

    X = torch.FloatTensor(x)
    Y = torch.FloatTensor(y)
    # print(X, Y)
    epsilon = 0.01
    niter = 500

    sinkhorn = SinkhornDistance(eps=0.01, max_iter=100)
    dist, P, C = sinkhorn(X, Y)
    
    inter = intersect(reachset, targetset)
    return dist.item(), inter

def metric(reachset, targetset, printC = False):
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
    
    inter = False
    if intersect(reachset, targetset):
        # to compute the intersection area
        overlapX = r_width + g_width - (max(reachset.x_sup, targetset.x_sup)-min(reachset.x_inf, targetset.x_inf))
        overlapY = r_height + g_height - (max(reachset.y_sup, targetset.y_sup)-min(reachset.y_inf, targetset.y_inf))
        assert overlapX >= 0
        assert overlapY >= 0
        area = overlapX*overlapY
        if printC:
            print("Ohoo, intersected and the area is: ", area)
        inter = True
        return -area, inter
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
        if printC:
            print("Not intersect and min distance is: ", min_dist)
        return min_dist, inter

def gradient(cmd, state, theta, goalset, step, measure, goal_run=True):
    gradient = []
    # x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]), str(theta[1]), str(85)], 
    #     stdout=subprocess.PIPE).stdout.decode('utf-8')
    # x = x.split("\n")
    # print(x)
    # x = [float(x[i]) for i in range(len(x)-1)]
    # reachset0 = set(x)
    # measure(reachset0, goalset)
    
    delta = np.random.uniform(low=-1, high=1, size=(theta.size))
    x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]+delta[0]), str(theta[1]), str(step)], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    reachset1 = set(x)

    x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]-delta[0]), str(theta[1]), str(step)], 
    stdout=subprocess.PIPE).stdout.decode('utf-8')
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    reachset2 = set(x)

    x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]), str(theta[1]+delta[1]), str(step)], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    reachset3 = set(x)

    x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]), str(theta[1]-delta[1]), str(step)], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    reachset4 = set(x)

    if goal_run:
        x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]), str(theta[1]), str(75)], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    else:
        x = subprocess.run([cmd, str(state[0]), str(state[1]), str(theta[0]), str(theta[1]), str(5)], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')        
    x = x.split("\n")
    x = [float(x[i]) for i in range(len(x)-1)]
    reachset5 = set(x)

    m1, _ = measure(reachset1, goalset)
    m2, _ = measure(reachset2, goalset)
    gradient.append(0.5*(m1-m2)/delta[0])

    m3, _ = measure(reachset3, goalset)
    m4, _ = measure(reachset4, goalset)
    gradient.append(0.5*(m3-m4)/delta[1])

    m5, inter = measure(reachset5, goalset, printC=True)
    gradient.append((reachset1.area-reachset2.area)/delta[0])
    gradient.append((reachset3.area-reachset4.area)/delta[1])

    print(gradient, m5, inter)

    return np.array(gradient), m5, inter


def run_heuristic():
    # here to parse the current state information.
    state_list = []
    theta_list = []
    disturbance = True
    
    env = ACC()
    goalset = set([145, 39, 155, 40])
    unsafeset = set([118, 25, 120, 55])

    cmd = './acc_model'

    lr = 0.0005
    indi = 10000
    print('here begins the program')
    for i in range(1):
        state = env.reset()
        # theta = np.random.uniform(low=0, high=1, size=(state.size))
        theta = np.array([0.5, -0.5])
        # theta = np.array([0.65948292, -2.37738382])
        # theta = np.array([0.62166829, -2.10912446])
        # theta = np.array([0.78902362, -3.08507751])
        better_theta = theta
        goal_dis = []
        safe_dis = []
        for t in range(200):
            current = time.time()
            print('') 
            print(t, theta, better_theta, indi)
            if t >= 100:
                lr = 0.0001 
            state_list.append(state)
            theta_list.append(theta)
            goal_gra, criterion, _ = gradient(cmd, np.array([123, 50]), theta, goalset=goalset, step=50, measure=metric)
            safe_gra, safe_distace, inter = gradient(cmd, np.array([123, 50]), theta=theta, goalset=unsafeset, step=6, goal_run=False, measure=metric)
            
            goal_dis.append(criterion)
            safe_dis.append(safe_distace)

            if criterion < 0 and safe_distace > 0:
                np.save('goal_acc.npy', np.array(goal_dis))
                np.save('safe_acc.npy', np.array(safe_dis))
                break
            # print(criterion, safe_distace)

            # assert False
            if criterion < indi and not inter:
                better_theta = theta 
                indi = criterion

            # print('goal_gra:' , goal_gra)
            
            if not disturbance:
                next_theta = theta - lr * goal_gra[:2] + 1e-4 * safe_gra[:2]
            else:
                next_theta = theta - lr * goal_gra[:2] + 1e-4 * safe_gra[:2]
                if criterion <= 0:
                    next_theta -= lr/5000 * goal_gra[2:]

            theta = next_theta
            # if t % 20 == 0:
            #     np.save('acc_theta_his_29.npy', np.array(theta_list))
            print('time: ', time.time()-current)
        print(better_theta)

def run_W():
    state_list = []
    theta_list = []
    
    env = ACC()
    goalset = set([145, 39, 155, 40])
    unsafeset = set([118, 25, 120, 55])

    cmd = './acc_model'

    lr = 0.00001
    indi = 10000
    print('here begins the program')
    for i in range(1):
        state = env.reset()
        theta = np.array([0.5, -0.5])
        # theta = np.array([0.65948292, -2.37738382])
        # theta = np.array([0.62166829, -2.10912446])
        # theta = np.array([0.86768563, -3.0358228])
        # theta = np.array([0.3162, -0.6789])
        better_theta = theta
        goal_dis = []
        safe_dis = []
        for t in range(200):
            current = time.time()
            print('') 
            print(t, theta, better_theta, indi)
            if t >= 100:
                lr = 0.0001 
            state_list.append(state)
            theta_list.append(theta)
            # _, _, _ = gradient(cmd, np.array([123, 50]), theta=theta, goalset=goalset, measure=W_distance, step=50)
            # assert False
            goal_gra, criterion, inter_goal = gradient(cmd, np.array([123, 50]), theta, goalset=goalset, measure=W_distance, step=50)
            safe_gra, safe_distace, inter_unsafe = gradient(cmd, np.array([123, 50]), theta=theta, goalset=unsafeset, measure=W_distance, step=5, goal_run=False)
            
            goal_dis.append(criterion)
            safe_dis.append(safe_distace)

            # if criterion < 0 and safe_distace > 0:
            #     np.save('goal_acc.npy', np.array(goal_dis))
            #     np.save('sage_acc.npy', np.array(safe_dis))
            # print(criterion, safe_distace)

            if inter_goal and not inter_unsafe:
                better_theta = theta
                np.save('goal_W_'+str(t)+'.npy', np.array(goal_dis)) 
                np.save('safe_W_'+str(t)+'.npy', np.array(safe_dis))
                break
            
            goal_gra = np.clip(goal_gra, -4*10**4, 4*10**4)
            next_theta = theta - lr * goal_gra[:2] + 1e-5 * safe_gra[:2]

            theta = next_theta
            print('time: ', time.time()-current)
        print(better_theta)

if __name__ == '__main__':
    # heuristic
    run_heuristic()

    # W_distance
    # run_W()