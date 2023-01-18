import base
from cvxopt import solvers, matrix
import numpy as np
from base import swarmbot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import imageio


def check_for_separation_dist(x, y, bot_positions):

    for set_pos in bot_positions:
        dist = ((set_pos[0] - x) ** 2 + (set_pos[1] - y) ** 2) ** 0.5
        if dist < base.seperation:
            return False
    return True


def initialize_swarm(num_of_bots = base.bots_count):

    # container to store the bot positions

    bots = []
    for i in range(num_of_bots):
        bot = swarmbot()
        bots.append(bot)
    bot_positions = []

    for i in range(len(bots)):
        pos_x, pos_y = np.random.normal(base.pos_mean, base.sd, size=(1, 2))[0]

        if check_for_separation_dist(pos_x, pos_y, bot_positions):
            bot_positions.append([pos_x, pos_y])
        else:
            dist_pass = False
            while not dist_pass:
                pos_x, pos_y = np.random.normal(base.pos_mean, base.sd, size=(1, 2))[0]
                dist_pass = check_for_separation_dist(pos_x, pos_y, bot_positions)

            bot_positions.append([pos_x, pos_y])

    for bot_, pos in zip(bots, bot_positions):
        bot_.q = np.array([[pos[0], pos[1]]])

    return bots


def goal_reached(current_state, desired_state):

    
    curr_centroid_x, curr_centroid_y, curr_theta, curr_s1, curr_s2 = current_state[0][0].item(), current_state[0][1].item(), current_state[1].item(), current_state[2], current_state[3]

    des_centroid_x, des_centroid_y, des_theta, des_s1, des_s2 = desired_state[0][0].item(), desired_state[0][1].item(), desired_state[1], desired_state[2],desired_state[3]

    # the logic is if the difference is non-ZERO in any case it means desired state is not reached yet.We round it off to two difits.
    if round(curr_centroid_x - des_centroid_x, ndigits=2):
        return False
    if round(curr_centroid_y - des_centroid_y, ndigits=2):
        return False
    if round(curr_theta - des_theta, ndigits=2):
        return False
    if round(curr_s1 - des_s1, ndigits=2):
        return False
    if round(curr_s2 - des_s2, ndigits=2):
        return False

    return True

    
def convex_optimizer(C, d, A, b):

    # in order to satisfy the inputs C and d of convex optimization equation
    C = np.sqrt(2) * C
    d = np.sqrt(2) * d

    # convert to cvxopt matrix
    C = matrix(C, C.shape, 'd')
    d = matrix(d, d.shape, 'd')

    A = matrix(A, A.shape, 'd')
    b = matrix(b, b.shape, 'd')


    P = C.T * C


    q = -d.T * C
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q.T, A, b)

    if 'x' in solution.keys():
        return solution['x']
    else:
        return np.array([[0], [0]])


def check_collision_avoidance(i, bots, R, u_curr, convergence_condition, convergence_constraint):

    # calculate bot position in moving frame
    first_bot_pos = np.matmul(R.transpose(), np.subtract(bots[i].q.transpose(), u_curr))

    # iterate through all the bots
    for j in range(len(bots)):
        if j == i:
            continue
        else:
            second_bot_pos = np.matmul(R.transpose(), np.subtract(bots[j].q.transpose(), u_curr))

            position_diff = np.subtract(first_bot_pos, second_bot_pos)

            delta = np.linalg.norm(position_diff, 2)

            # collision avoidance condition
            if delta <= base.seperation:

                first_bot_vel = np.matmul(R.transpose(), bots[i].vel.transpose())

                second_bot_vel = np.matmul(R.transpose(), bots[j].vel.transpose())

                velo_diff = np.subtract(first_bot_vel, second_bot_vel)

                collision_avoidance_condition = np.matmul(position_diff, velo_diff.transpose())

                convergence_condition = np.vstack((convergence_condition, -collision_avoidance_condition))
                convergence_constraint = np.vstack((convergence_constraint, np.array([[0.0], [0.0]])))

    return convergence_condition, convergence_constraint

def draw_ellipse(centroid, s1, s2, orientation):
    return Ellipse(xy=(centroid[0], centroid[1]), width=base.concentrated_ellipse * s1, height=base.concentrated_ellipse * s2, angle=orientation * 180/np.pi, edgecolor='b', fill=False)


def plot_swarm(bots, current_state, goal_state, count):
    centroid = current_state[0]
    orientation = current_state[1]
    s1 = current_state[2]
    s2 = current_state[3]
    centroid_g = goal_state[0]
    orientation_g = goal_state[1]
    s1_g = goal_state[2]
    s2_g = goal_state[3]

    results = "./results"
    if not os.path.exists(results):
        os.mkdir(results)
    plt.figure()
    ax = plt.gca()
    ellipse1 = draw_ellipse(centroid, s1, s2, orientation)
    ellipse2 = draw_ellipse(centroid_g, s1_g, s2_g, orientation_g)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    plt.xlim((-20, 20))
    plt.ylim((-20, 20))

    for bot in bots:
        bot_pos = bot.q.T
        circle = plt.Circle((bot_pos[0], bot_pos[1]), (base.axle_len+base.rad), color='r', fill=False)
        ax.add_patch(circle)

        x, y = bot_pos[0], bot_pos[1]
        length = (base.axle_len+base.rad)
        orientation = bot.theta
        endy = y + length * np.sin(orientation)
        endx = x + length * np.cos(orientation)

        plt.plot([x, endx], [y, endy])

    # if draw_lines is not None:
    #     plt.plot(draw_lines[0][0], draw_lines[1][0], color='r', markersize=3)
    #     plt.plot(draw_lines[0][1], draw_lines[1][1], color='r', markersize=3)

    file_path = os.path.join(results, str(count) + ".png")
    plt.savefig(file_path)
    plt.show()



def results(state_pts, vel_cap, vel_star, last_lin_vel, last_ang_vel):

    state_tld_pts_x = [state_tld[0].item() for state_tld in state_pts]
    state_tld_pts_y = [state_tld[1].item() for state_tld in state_pts]
    state_tld_pts_theta = [state_tld[2].item() for state_tld in state_pts]
    state_tld_pts_s1 = [state_tld[3].item() for state_tld in state_pts]
    state_tld_pts_s2 = [state_tld[4].item() for state_tld in state_pts]

    vel_star_all_x = []
    vel_star_all_y = []
    for i in range(len(vel_star[0])):
        vel_star_all_x.append([])
        vel_star_all_y.append([])
    for i in range(len(vel_star)):
        vel_all_bots = vel_star[i]
        for j in range(len(vel_all_bots)):
            vel_star_all_x[j].append(vel_all_bots[j][0])
            vel_star_all_y[j].append(vel_all_bots[j][1])

    vel_cap_all_x = []
    vel_cap_all_y = []
    for i in range(len(vel_cap[0])):
        vel_cap_all_x.append([])
        vel_cap_all_y.append([])
    for i in range(len(vel_cap)):
        vel_all_bots = vel_cap[i]
        for j in range(len(vel_all_bots)):
            vel_cap_all_x[j].append(vel_all_bots[j][0])
            vel_cap_all_y[j].append(vel_all_bots[j][1])

    plt.figure()
    plt.ylim((-10, 10))
    for i in range(len(vel_star_all_x)):
        plt.plot(vel_star_all_x[i], '-')
        plt.plot(vel_cap_all_x[i], '--')
    plt.legend(loc='upper right')
    plt.title("u_ix and u_star_ix vs time")
    plt.ylabel("u_ix and u_star_ix")
    plt.xlabel("time")
    plt.savefig('results/optimal_computed_velocity_x.jpg')

    plt.figure()


    plt.ylim((-10, 10))
    for i in range(len(vel_star_all_y)):
        plt.plot(vel_star_all_y[i], '-')
        plt.plot(vel_cap_all_y[i], '--')
    plt.legend(loc='upper right')
    plt.title("u_iy and u_star_iy vs time")
    plt.ylabel("u_iy and u_star_iy")
    plt.xlabel("time")
    plt.savefig('results/optimal_computed_velocity_y.jpg')

    plt.figure()
    plt.ylim((-0.5, 0.5))
    plt.plot(last_lin_vel)
    plt.title("linear velocity vs time")
    plt.xlabel("time")
    plt.ylabel("linear velocity")
    plt.savefig('results/linear_velocity.jpg')

    plt.figure()
    plt.ylim((-2, 2))
    plt.plot(last_ang_vel)
    plt.title("angular velocity vs time")
    plt.xlabel("time")
    plt.ylabel("angular velocity")
    plt.savefig('results/angular_velocity.jpg')

    plt.figure()
    plt.title("desired state vs time")
    plt.xlabel("time")
    plt.ylabel("desired states")
    plt.plot(state_tld_pts_x, label='~x or x_des')
    plt.plot(state_tld_pts_y, label='~y or y_des')
    plt.plot(state_tld_pts_theta, label='~theta or theta_des')
    plt.plot(state_tld_pts_s1, label='~s1 or s1_des')
    plt.plot(state_tld_pts_s2, label='~s2 or s2_des')
    plt.legend(loc='upper right')
    plt.savefig('results/state_tilde_plot.jpg')

    plt.show()

def simulate():
    png_dir = './results'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file))

    for _ in range(100):
        images.append(imageio.imread(file))

    imageio.mimsave('./results/2d_simulation.gif', images)
