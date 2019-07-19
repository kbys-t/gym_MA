# coding:utf-8
"""
Move to goal with formation
Goal moves along with sine curve
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class PersonalSpaceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.DT = 0.02
        self.ENV_MAX_X = 2.0
        self.ENV_MAX_Y = 2.0
        self.RESTITUTION = 0.5

        # Limitation
        self.MAX_ANG_VEL = [6.0, 10.0]
        self.MAX_ANG_ACC = [6.0, 10.0]
        self.CIRCLE_RADIUS = 1.5
        self.MAX_ARC = self.CIRCLE_RADIUS * np.pi

        # for multi agents
        self.AGE_NUM = 2
        self.AGE_TASK = [0, 1]
        self.AGE_GAIN = [1.0, 1.0]
        self.AGE_SIZE = [0.2, 0.2]
        self.AGE_INI_ANG = [[0.0], [0.8*np.pi]]
        self.TARGET_DIST = self.CIRCLE_RADIUS * 0.2 * np.pi

        self.DIFF_THETA_LIM = 2.0 * np.arcsin(0.5*np.sum(self.AGE_SIZE)/self.CIRCLE_RADIUS)
        # Create spaces
        self.action_space = []
        self.observation_space = []
        for i in range(self.AGE_NUM):
            high_a = np.array([self.MAX_ANG_ACC[i]])
            # high_s = np.array([1.0, 1.0, self.MAX_ANG_VEL[i], self.MAX_ARC])
            high_s = np.array([1.0, 1.0, self.MAX_ANG_VEL[i], 1.0, 1.0])
            self.action_space.append(spaces.Box(-high_a, high_a))
            self.observation_space.append(spaces.Box(-high_s, high_s))

        # Initialize
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = []
        for i in range(self.AGE_NUM):
            self.state.append([self.AGE_INI_ANG[i][0], 0.0, 0.0, 0.0])
        for i in range(self.AGE_NUM):
            for j in range(self.AGE_NUM):
                if not j==i:
                    self.state[i][2] = self._calc_dist([self.state[i][0], self.state[j][0]])
                    self.state[i][3] = self._angle_normalize(self.state[i][0] - self.state[j][0])
        self.state = np.array(self.state)
        return self._get_obs()

    def step(self, action):
        s = self.state
        a = np.array([ [np.clip(action[i][0], -self.MAX_ANG_ACC[i], self.MAX_ANG_ACC[i])] for i in range(self.AGE_NUM) ])

        # update the position of each agent
        ns = self._dynamics(s, a, self.DT)
        for i in range(self.AGE_NUM):
            self.state[i] = ns[i]

        # reward design
        reward = self._get_reward(self.AGE_TASK, self.AGE_GAIN, self.state, a)

        done = False

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        s = self.state
        rtv = []
        for i in range(self.AGE_NUM):
            rtv.append(np.array([np.sin(s[i][0]), np.cos(s[i][0]), s[i][1], np.sin(s[i][3]), np.cos(s[i][3])]))
        return rtv

    def _dynamics(self, s, a, dt):
        ns = s
        flag_coll = False
        for i in range(self.AGE_NUM):
            ar = a[i,0]
            ns[i,1] += ar * dt
            ns[i,1] = np.clip(ns[i,1], -self.MAX_ANG_VEL[i], self.MAX_ANG_VEL[i])
            ns[i,0] += ns[i,1] * dt
            ns[i,0] = self._angle_normalize(ns[i][0])

        for i in range(self.AGE_NUM):
            for j in range(self.AGE_NUM):
                if not j==i:
                    if np.abs(self._angle_normalize(ns[i,0]-ns[j,0]))<self.DIFF_THETA_LIM:
                        over = self.DIFF_THETA_LIM - np.abs(self._angle_normalize(ns[i,0]-ns[j,0])) #>0
                        flag_coll = True
                        break

        if flag_coll:
            if np.sign(ns[0,1])==-np.sign(ns[0,2]):
                ns[1,0] = ns[1,0] - np.sign(ns[1,2]) * over
                ns[1,1] *= -self.RESTITUTION
            elif np.sign(ns[1,1])==-np.sign(ns[1,2]):
                ns[0,0] = ns[0,0] - np.sign(ns[0,2]) * over
                ns[0,1] *= -self.RESTITUTION
            else:
                deno = np.abs(ns[0,1]) + np.abs(ns[1,1])
                for i in range(self.AGE_NUM):
                    ns[i,0] = ns[i,0] - np.sign(ns[i,2]) * over * np.abs(ns[i,1]) / deno
                    ns[i,1] *= -self.RESTITUTION
                    ns[i,0] = self._angle_normalize(ns[i,0])


        for i in range(self.AGE_NUM):
            for j in range(self.AGE_NUM):
                if not j==i:
                    ns[i,2] = self._calc_dist([ns[i,0], ns[j,0]])
                    ns[i,3] = self._angle_normalize(ns[i,0] - ns[j,0])

        # for i in range(self.AGE_NUM):
        #     if np.abs(ns[i,1])<np.sum(self.AGE_SIZE):
        #         over = np.abs( np.abs(ns[i,1]) - np.sum(self.AGE_SIZE) )
        #         flag_coll = True
        #         break

        return ns

    def _get_reward(self, num, g, s, a):
        rtv = np.zeros((self.AGE_NUM))
        for i in range(self.AGE_NUM):
            if num[i]==0:
                # idx = np.where(dist[i]==np.min([dist[i] for i in range(self.AGE_NUM) if not j==i]))[0][0]
                rtv[i] = 2.0 * np.exp( -0.7 * np.abs(np.abs(s[i,2]) - np.sum(self.AGE_SIZE)) ) - 1.0
                # rtv[i] = 1.0 - 2.0 * np.abs(np.abs(s[i,2]) - np.sum(self.AGE_SIZE)) / ( np.pi * self.CIRCLE_RADIUS - np.sum(self.AGE_SIZE))
            else:
                # idx = np.where(dist[i]==np.min([dist[i,j] for i in range(self.AGE_NUM) if not j==i]))[0][0]
                rtv[i] = 2.0 * np.exp( -0.7 * np.abs( np.abs(np.abs(s[i,2]) - np.sum(self.AGE_SIZE)) - self.TARGET_DIST ) ) - 1.0
                # rtv[i] = 1.0 - 2.0 * np.abs(np.abs(s[i,2]) - np.sum(self.AGE_SIZE) - self.TARGET_DIST) / ( np.pi * self.CIRCLE_RADIUS - np.sum(self.AGE_SIZE) - self.TARGET_DIST)
        return rtv

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)


    def _calc_dist(self, x):
        diff_theta = self._angle_normalize(x[0]-x[1])
        rtv = -self.CIRCLE_RADIUS * diff_theta
        return rtv

    def render(self, mode='human'):
        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(200*self.ENV_MAX_X/self.ENV_MAX_Y),200)
            self.viewer.set_bounds(-self.ENV_MAX_X,self.ENV_MAX_X,-self.ENV_MAX_Y,self.ENV_MAX_Y)

        pers = self.viewer.draw_circle(np.sum(self.AGE_SIZE)+self.TARGET_DIST)
        pers.set_color(185.0/255.0*255.0/185.0, 134.0/255.0*255.0/185.0, 164.0/255.0*255.0/185.0)
        pers_trans = rendering.Transform(rotation=0.0, translation=[self.CIRCLE_RADIUS*np.cos(s[1,0]), self.CIRCLE_RADIUS*np.sin(s[1,0])])
        pers.add_attr(pers_trans)

        rail = self.viewer.draw_circle(self.CIRCLE_RADIUS, filled=False)
        rail_trans = rendering.Transform(rotation=0.0, translation=[0.0, 0.0])
        rail.add_attr(rail_trans)

        for i in range(self.AGE_NUM):
            ages = self.viewer.draw_circle(self.AGE_SIZE[i])
            if i==0:
                ages.set_color(22.0/255.0, 108.0/255.0, 156.0/255.0)
            elif i==1:
                ages.set_color(185.0/255.0, 134.0/255.0, 164.0/255.0)
            age_trans = rendering.Transform(rotation=0.0, translation=[self.CIRCLE_RADIUS*np.cos(s[i,0]), self.CIRCLE_RADIUS*np.sin(s[i,0])])
            ages.add_attr(age_trans)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
