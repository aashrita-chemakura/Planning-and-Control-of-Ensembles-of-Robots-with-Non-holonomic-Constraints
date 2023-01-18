import base
from base import abs_space
import utils
import numpy as np


def main_algo(num_bots=None, num_iterations=None):
    if num_bots is None:
        num_bots = base.bots_count
    if num_iterations is None:
        num_iterations = base.iters

    abstract_space = abs_space()
    curr_centroid = abstract_space.get_centroid(None)
    theta_curr, s1_curr, s2_curr = abstract_space.parameters(None)

    bots = utils.initialize_swarm(num_bots)

    counter = 0
    vel_cap_pts = []
    vel_star_pts = []
    bot_last_vel_pts_lin = []
    bot_last_vel_pts_ang = []
    state_tilde_plot_pts = []

    desired_state_index = 0

    while True:

        curr_centroid = abstract_space.get_centroid(bots)
        theta_curr, s1_curr, s2_curr = abstract_space.parameters(bots)
        current_state = [curr_centroid, theta_curr, s1_curr, s2_curr]

        desired_state = base.des_abstract_state[desired_state_index]

        if counter % 1000 == 0:
            utils.plot_swarm(bots, current_state,   desired_state, str(desired_state_index) +"_" + str(counter))

        if utils.goal_reached(current_state, desired_state) \
                or counter > num_iterations:
            if len(base.des_abstract_state) > 1 and desired_state_index < len(base.des_abstract_state) - 1:
                desired_state_index += 1
                counter = 0
            else:
                break


        state_tilde = np.vstack((np.subtract(desired_state[0], current_state[0]),
                                    desired_state[1] - current_state[1],
                                    desired_state[2] - current_state[2],
                                    desired_state[3] - current_state[3]))


        control_vector_fields = [np.matmul(base.KU, np.vstack((state_tilde[0], state_tilde[1]))), base.KT*state_tilde[2].item(), base.KS1*state_tilde[3].item(), base.KS2*state_tilde[4].item()]
        centroid_derivative, theta_derivative, s1_derivative, s2_derivative = control_vector_fields[0],\
                                                                          control_vector_fields[1],\
                                                                          control_vector_fields[2],\
                                                                          control_vector_fields[3]
        R, H1, H2, H3 = abstract_space.formation_variables(theta_curr)

        Lie_grp = np.vstack(((np.hstack((R, curr_centroid))), np.array([0, 0, 1])))

        Gamma = np.vstack((np.hstack((Lie_grp, np.zeros(shape=(3, 2)))),
                           np.hstack((np.zeros(shape=(2, 3)), base.I))))

        vel_cap_all_bots = []
        vel_star_all_bots = []

        for i in range(len(bots)):

            bot = bots[i]

            vel_star = np.add(np.add(centroid_derivative, (
                                    (s1_curr - s2_curr) * np.matmul(H3, np.subtract(bot.q.transpose(), curr_centroid)) * theta_derivative / (s1_curr + s2_curr))),
                                    np.add((np.matmul(H1, np.subtract(bot.q.transpose(), curr_centroid)) * s1_derivative / 4 * s1_curr),
                                    (np.matmul(H2, np.subtract(bot.q.transpose(), curr_centroid)) * s2_derivative / 4 * s2_curr)))

            bot_pos_moving_frame = np.matmul(R.transpose(), np.subtract(bot.q.transpose(), curr_centroid))

            # calculate differential of surjective submersion
            diff_phi = np.vstack((base.I,
                             (1/s1_curr-s2_curr)*np.matmul(bot_pos_moving_frame.transpose(), base.E1),
                              np.matmul(bot_pos_moving_frame.transpose(), np.add(base.I, base.E2)),
                              np.matmul(bot_pos_moving_frame.transpose(), np.subtract(base.I, base.E2))))

            # monotonic convergence constraint
            convergence_condition = np.matmul(np.matmul(np.matmul(state_tilde.transpose(),
                                                                  base.gain_matrix), Gamma), diff_phi)

            # flip the sign to maintain the inequality given at (14) of [1] for convex optimization
            convergence_condition = -convergence_condition
            convergence_constraint = np.array([[0.0]])


            # check for collision avoidance with every team member and impose additional constraint if neccessary
            convergence_condition, convergence_constraint = utils.check_collision_avoidance(i, bots, R, curr_centroid,
                                                                                            convergence_condition,
                                                                                            convergence_constraint)

            try:
                vel_convex_opt = utils.convex_optimizer(base.I, np.matmul(R.transpose(), vel_star),
                                                        convergence_condition, convergence_constraint)
            except Exception as e:
                break

            if counter == 0 or counter % 1000 == 0:
                vel_cap_all_bots.append(vel_convex_opt)
                vel_star_all_bots.append(vel_star)

            vel_inertial_frame = np.matmul(R, vel_convex_opt)

            bots[i].move_bot(vel_inertial_frame_cvxopt=vel_inertial_frame,
                                  vel_inertial_frame_optimal=vel_star.transpose())

        # i = 0
        if counter % 50 == 0:
            state_tilde_plot_pts.append(state_tilde)
            bot_last_vel_pts_lin.append(bots[0].linear_vel)
            bot_last_vel_pts_ang.append(bots[0].angular_vel)
        if counter % 1000 == 0:
            vel_cap_pts.append(vel_cap_all_bots)
            vel_star_pts.append(vel_star_all_bots)

        counter = counter + 1

    utils.results(state_tilde_plot_pts, vel_cap_pts, vel_star_pts, bot_last_vel_pts_lin, bot_last_vel_pts_ang)

    utils.simulate()

main_algo()
