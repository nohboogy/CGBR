import numpy as np
import os
from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def goal_square_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    distance = np.abs(goal_a - goal_b)
    return distance


class ManiEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper, holding_block,
            has_object, target_in_the_air, target_offset, obj_range, target_range, fixed_range,
            distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.holding_block = holding_block
        self.has_object = has_object
        self.fixed_range = fixed_range
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.count = 3
        self.contact_check = 0  # no contact
        self.contact_pos = 0
        self.data = []
        super(ManiEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, threshold):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        if self.holding_block:
            # self.set_box_geom(threshold)
            reward = (d < threshold).astype(np.float32) - 1
            if len(achieved_goal.shape) == 1:
                if abs(achieved_goal[2] - goal[2]) > 0.2:
                    reward = -1
                return reward
            else:
                for i in range(achieved_goal.shape[0]):
                    if reward[i] == 0:
                        if abs(achieved_goal[i, 2] - goal[i, 2]) > 0.2:
                            reward[i] = -1
                return reward
        else:
            if self.reward_type == 'sparse':
                return -(d > threshold).astype(np.float32)
            else:
                return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.5  # limit maximum change in position
        rot_ctrl = [0., 0, 1, 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        gripper_pos = self.sim.data.get_site_xpos('robot0:grip')
        z_move = gripper_pos[2] + action[2]
        if z_move < 0.15:
            action[2] = 0.15 - gripper_pos[2]

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        """
        joint_qpos = np.zeros(9)
        joint_qvel = np.zeros(9)
        #joint qpos  #2, 3, 5 joints have opposite axis
        joint_qpos[0] = self.sim.data.get_joint_qpos('robot0:joint1')
        joint_qpos[1] = -self.sim.data.get_joint_qpos('robot0:joint2')
        joint_qpos[2] = -self.sim.data.get_joint_qpos('robot0:joint3')
        joint_qpos[3] = self.sim.data.get_joint_qpos('robot0:joint4')
        joint_qpos[4] = -self.sim.data.get_joint_qpos('robot0:joint5')
        joint_qpos[5] = self.sim.data.get_joint_qpos('robot0:joint6')
        joint_qpos[6] = self.sim.data.get_joint_qpos('robot0:Gripper_base')
        joint_qpos[7] = self.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')
        joint_qpos[8] = self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')

        joint_qvel[0] = self.sim.data.get_joint_qvel('robot0:joint1')
        joint_qvel[1] = -self.sim.data.get_joint_qvel('robot0:joint2')
        joint_qvel[2] = -self.sim.data.get_joint_qvel('robot0:joint3')
        joint_qvel[3] = self.sim.data.get_joint_qvel('robot0:joint4')
        joint_qvel[4] = -self.sim.data.get_joint_qvel('robot0:joint5')
        joint_qvel[5] = self.sim.data.get_joint_qvel('robot0:joint6')
        joint_qvel[6] = self.sim.data.get_joint_qvel('robot0:Gripper_base')
        joint_qvel[7] = self.sim.data.get_joint_qvel('robot0:l_gripper_finger_joint')
        joint_qvel[8] = self.sim.data.get_joint_qvel('robot0:r_gripper_finger_joint')

        joint_data = np.concatenate([joint_qpos, joint_qvel], axis=0)
        self.data.append(joint_data)
        """
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        elif self.holding_block:
            if self.contact_check == 0:
                for i in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[i]
                    geom1 = self.sim.model.geom_id2name(contact.geom1)
                    geom2 = self.sim.model.geom_id2name(contact.geom2)
                    if geom1 == 'object0' or geom1 == 'table0':
                        if geom2 == 'object0' or geom2 == 'table0':
                            self.contact_check = 1  # contact
                            self.contact_pos = contact.pos.copy()
                if self.contact_check == 1:
                    achieved_goal = self.contact_pos
                else:
                    achieved_goal = np.squeeze(object_pos.copy())
            else:
                achieved_goal = self.contact_pos

        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:Gripper_base')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 25
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        """
        if self.holding_block:
            box_id = self.sim.model.body_name2id('container_box')
            self.sim.model.body_pos[box_id] = self.goal

        else:
            self.sim.model.site_pos[site_id] = self.goal
        """
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        """
        if self.data is not None:
            data_out = np.asarray(self.data.copy())
            np.savetxt('data.txt', data_out, delimiter=',')
        """
        self.sim.set_state(self.initial_state)
        self.contact_check = 0  # no contact
        self.contact_pos = 0
        # Randomize start position of object.
        if self.has_object:
            if self.holding_block:
                # Move end effector into position.
                gripper_target = self.initial_gripper_xpos + self.np_random.uniform(-1, 1, size=3)
                gripper_rotation = np.array([0., 0., 1., 0.])
                self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                for _ in range(10):
                    self.sim.step()
                object_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                object_qpos[:3] = object_xpos
                object_qpos[2] += 0.15
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)
                # self.sim.data.set_joint_qpos('robot0:joint6',  np.pi / 2 + np.random.rand(1)/10)
            else:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.holding_block:
                if self.target_in_the_air and self.np_random.uniform() < 4:
                    goal[2] += self.np_random.uniform(0, 3.95)

            elif self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal, threshold):
        # self.set_box_geom(threshold)
        d = goal_distance(achieved_goal, desired_goal)
        return (d < threshold).astype(np.float32)
        """
        if self.holding_block:
            success = (d < threshold).astype(np.float32)
            if len(achieved_goal.shape) == 1:
                if not 0 <= achieved_goal[2] - desired_goal[2] <= 0.5:
                    success = 0
                return success
            else:
                for i in range(achieved_goal.shape[0]):
                    if success[i] == 1:
                        if not 0 <= achieved_goal[i, 2] - desired_goal[i, 2] <= 0.5:
                            success[i] = 0
                return success
        else:
            return (d < threshold).astype(np.float32)
        """

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        if not self.holding_block:
            # Move end effector into position.
            gripper_target = self.sim.data.get_site_xpos('robot0:grip')
            gripper_rotation = np.array([0., 0., 1., 0.])
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            for _ in range(10):
                self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        if self.holding_block:
            self.height_offset = 0.15
            # self.box_geom_id = np.zeros(5, dtype=np.int)
            # self.box_geom_id[0] = self.sim.model.geom_name2id('cbox_down')
            # self.box_geom_id[1] = self.sim.model.geom_name2id('cbox_up')
            # self.box_geom_id[2] = self.sim.model.geom_name2id('cbox_left')
            # self.box_geom_id[3] = self.sim.model.geom_name2id('cbox_right')
            # self.box_geom_id[4] = self.sim.model.geom_name2id('cbox_floor')


"""
    def set_box_geom(self, threshold):
        self.sim.model.geom_size[self.box_geom_id[0]] = np.array([1 * threshold + 0.1, 0.1, 0.25])
        self.sim.model.geom_pos[self.box_geom_id[0]] = np.array([-0.1, -(1 * threshold + 0.1), 0.25])

        self.sim.model.geom_size[self.box_geom_id[1]] = np.array([1 * threshold + 0.1, 0.1, 0.25])
        self.sim.model.geom_pos[self.box_geom_id[1]] = np.array([0.1, 1 * threshold + 0.1, 0.25])

        self.sim.model.geom_size[self.box_geom_id[2]] = np.array([0.1, 1 * threshold + 0.1, 0.25])
        self.sim.model.geom_pos[self.box_geom_id[2]] = np.array([-(1 * threshold + 0.1), 0.1, 0.25])

        self.sim.model.geom_size[self.box_geom_id[3]] = np.array([0.1, 1 * threshold + 0.1, 0.25])
        self.sim.model.geom_pos[self.box_geom_id[3]] = np.array([1 * threshold + 0.1, -0.1, 0.25])

        self.sim.model.geom_size[self.box_geom_id[4]] = np.array([1 * threshold + 0.2, 1 * threshold + 0.2, 0.1])
"""
