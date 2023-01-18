import numpy as np
import math
from math import *


iters = 15000 #iterations

bots_count = 5 #number of bots

pos_mean = 8.5 #mean

sd = 1 #standard deviation

del_t = 0.01 #time difference

# gain matrices 
KU = np.array([[2, 0], [0, 2]])
KS1 = 2
KS2 = 2
KT = 2

# formation variables 
E1 = np.array([[0.0, 1.0], [1.0, 0.0]])
E2 = np.array([[1.0, 0.0], [0.0, -1.0]])


# robot specifications
rad = 0.15
axle_len = 0.1
safe_dist = 0.1
max_lin_vel = 0.3
max_ang_vel = 1

# separation distance
seperation = 2*(rad + axle_len) + safe_dist

# calculate concentration ellipse
prob = 0.99
concentrated_ellipse = -2*math.log(1-prob)

# desired state space
# desired centroid
u_des = np.array([[10], [6]], dtype=np.float32)

# desired orientation
theta_des = 0.0

# desired semi major axis of the ellipse
s1_des = 0.8

# desired semi minor axis of the ellipse
s2_des = 0.6

# desired state space
des_abstract_state = [[u_des, theta_des, s1_des, s2_des]]

# Gain matrix K
gain_matrix = np.vstack((np.array([1, 0, 0, 0, 0]),
                     np.array([0, 1, 0, 0, 0]),
                     np.array([0, 0, 0.8, 0, 0]),
                     np.array([0, 0, 0, 0.8, 0]),
                     np.array([0, 0, 0, 0, 0.8])))

I = np.eye(2)


class abs_space:

    def __init__(self):

        self.u_curr = np.zeros(shape=[2, 1], dtype=np.float32)
        self.theta_curr = 0.0
        self.s1_curr = 0.0
        self.s2_curr = 0.0

    def formation_variables(self, theta):

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H1 = np.eye(2) + np.matmul(np.matmul(R, R), E2)
        H2 = np.eye(2) - np.matmul(np.matmul(R, R), E2)
        H3 = np.matmul(np.matmul(R, R), E1)

        return R, H1, H2, H3
   
    
    def get_centroid(self, swarm_bots=None):

        if swarm_bots is None:
            return self.u_curr
        bot_centroids = 0.0
        for bot in swarm_bots:
            bot_centroids = bot_centroids + bot.q.transpose()
    

        return bot_centroids / len(swarm_bots)
    
    
    
    def parameters(self, swarm_bots):
        abstract_space_x = 0.0
        abstract_space_y = 0.0
        abstract_space_s1 = 0.0
        abstract_space_s2 = 0.0
        
        if swarm_bots is None:
            return self.theta_curr, self.s1_curr, self.s2_curr

        self.u_curr = self.get_centroid(swarm_bots)
        for bot in swarm_bots:
            bot_pos = bot.q.transpose()
            abstract_space_y = abstract_space_y + np.matmul(np.matmul((bot_pos - self.u_curr).transpose(), E1),
                                     (bot.q.transpose() - self.u_curr)).item()
            abstract_space_x =  abstract_space_x + np.matmul(np.matmul((bot_pos - self.u_curr).transpose(), E2),
                                     (bot.q.transpose() - self.u_curr)).item()
            self.theta_curr = np.arctan2(abstract_space_y, abstract_space_x) / 2.0
            _, H1, H2, _ = self.formation_variables(self.theta_curr)

            abstract_space_s1 += np.matmul(np.matmul((bot_pos - self.u_curr).transpose(), H1),
                                (bot.q.transpose() - self.u_curr)).item()
            abstract_space_s2 += np.matmul(np.matmul((bot_pos - self.u_curr).transpose(), H2),
                                (bot.q.transpose() - self.u_curr)).item()
            
    
            self.s1_curr = abstract_space_s1 / (2 * (len(swarm_bots) - 1))
            self.s2_curr = abstract_space_s2 / (2 * (len(swarm_bots)  - 1))
            return self.theta_curr, self.s1_curr, self.s2_curr


class swarmbot:
    def __init__(self):

        self.q = np.zeros(shape=[1, 2], dtype=np.float32)
        self.vel = np.zeros(shape=[1, 2], dtype=np.float32)
        self.vel_star = np.zeros(shape=[1, 2], dtype=np.float32)
        self.theta = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0

    def move_bot(self, vel_inertial_frame_cvxopt, vel_inertial_frame_optimal): #updates tbot position and orientation

        transform_matrix = np.vstack(([np.cos(self.theta), np.sin(self.theta)],
                                 [-np.sin(self.theta)/axle_len, np.cos(self.theta)/axle_len]))

        vel_moving_frame = np.matmul(transform_matrix, vel_inertial_frame_cvxopt)

        (self.linear_vel, self.angular_vel)  = (min(vel_moving_frame[0].item(), max_lin_vel), min(vel_moving_frame[1].item(), max_ang_vel))


        linear_disp = del_t * self.linear_vel * np.array([[np.cos(self.theta), np.sin(self.theta)]])

        angular_disp = del_t * self.angular_vel

        self.q=np.add(self.q, linear_disp)
        self.vel=vel_inertial_frame_cvxopt.transpose()
        self.vel_star=vel_inertial_frame_optimal
        self.theta=self.theta + angular_disp
