# This file define the Oscillator dynamics, reward function and safety property
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import random
import math
import gym
from interval import Interval

def main():
    trajectory = []
    env = ACC()
    initstate = []
    for ep in range(1):
        state = env.reset(122, 52)
        initstate.append(state)
        for i in range(200):
            trajectory.append(state)
            # 0.7699509  -2.76320723  0.48737962 safe [153.2, 39.96]
            # 0.66791895 -2.46594991  0.49806706 unsafe [158.7, 39.95]
            # control_input = 0.7699509  * state[0] - 2.76320723   * state[1] + 0.487379
            control_input = 0.80229365*state[0] -2.86548463 * state[1]
            # control_input = 0.71968267*state[0] -2.38933787 * state[1]


            # from ICCAD
            # control_input = 1.934 * (state[0]-150) - 2.6622 * (state[1]-40) + 8
            next_state, reward, done = env.step(control_input)
            print(i, state, control_input, next_state, reward, done)
            # if state[0] in Interval(139.5, 140.5) and state[1] in Interval(35.5, 36):
            #     print('unsafe')
            #     break
            state = next_state
        if i < 199:
            print(i, initstate[ep])

    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.savefig('ACC_trajectory.pdf', bbox_inches='tight')

class ACC:
    deltaT = 0.1
    max_iteration = 200
    error = 1e-5
    x0_low = 122
    x0_high = 124 
    x1_low = 48 
    x1_high = 52
    def __init__(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=self.x0_low, high=self.x0_high, size=1)[0]
            x1 = np.random.uniform(low=self.x1_low, high=self.x1_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
        else:
            self.x0 = x0
            self.x1 = x1
        
        self.t = 0
        self.state = np.array([self.x0, self.x1])
        self.u_last = 0

    def reset(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=self.x0_low, high=self.x0_high, size=1)[0]
            x1 = np.random.uniform(low=self.x1_low, high=self.x1_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
        else:
            self.x0 = x0
            self.x1 = x1
        
        self.t = 0
        self.state = np.array([self.x0, self.x1])
        self.u_last = 0
        return self.state

    def step(self, action):
        u = action 
        # x0_tmp = self.state[0] - self.deltaT * (self.state[1] - 40)
        # x1_tmp = self.state[1] - self.deltaT * (0.2 * self.state[1] - u)
        disturbance = np.random.uniform(low=-0.05, high=0.05, size=(1,))[0] * u
        # disturbance = 0
        # print(u, disturbance)
        # assert False
        x0_tmp = self.state[0] + 4 - 0.099 * self.state[1]
        x1_tmp = 0.9802 * self.state[1] + 0.1 * (u + disturbance)
        self.t = self.t + 1
        reward = self.design_reward(u, self.u_last, smoothness=1)
        self.u_last = u
        self.state = np.array([x0_tmp, x1_tmp])
        done = self.if_unsafe() or self.t == self.max_iteration
        return self.state, reward, done

    def design_reward(self, u, u_last, smoothness):
        r = 0
        r -= 1 / smoothness * abs(self.state[0])
        r -= 1 / smoothness * abs(self.state[1])
        r -= smoothness * abs(u)
        r -= smoothness * abs(u - u_last)
        if self.if_unsafe():
            r -= 50
        else:
            r += 10        
        return r

    def if_unsafe(self):
        if self.state[0] in Interval(self.x0_low, self.x0_high) and self.state[1] in Interval(self.x1_low, self.x1_high):
            return 0
        else:
            return 1


if __name__ == '__main__':
    main()