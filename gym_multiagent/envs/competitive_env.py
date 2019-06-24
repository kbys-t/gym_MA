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

class CompetitiveEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.DT = 0.02

        # Limitation
        self.MAX_VEL = 5.0
        self.MAX_BALL_VEL = 1.5
        self.MAX_POS_X = 2.0
        self.MAX_POS_Y = 1.0

        # Physical params
        self.OBJ_RESTITUTION = 0.8
        self.OBJ_SIZE = 0.06

        # for multi agents
        self.AGE_NUM = 4
        self.AGE_TASK = [0, 1, 2, 3]
        self.AGE_GAIN = [1.0, 1.0, 1.0, 1.0]
        self.AGE_SIZE = [[0.1,0.5],[0.1,0.5],[0.1,0.1],[0.1,0.1]]
        self.AGE_MAX_POS = [[-self.MAX_POS_X+0.0,-self.MAX_POS_X/2.0],\
            [-self.MAX_POS_X/2.0,0.0],\
            [0.0,self.MAX_POS_X/2.0],\
            [self.MAX_POS_X/2.0,self.MAX_POS_X-0.0]]
        self.AGE_INI_POS = [[(self.AGE_MAX_POS[1][0]+self.AGE_MAX_POS[1][1])/2.0,0.0],\
            [(self.AGE_MAX_POS[1][0]+self.AGE_MAX_POS[1][1])/2.0,0.0],\
            [(self.AGE_MAX_POS[2][0]+self.AGE_MAX_POS[2][1])/2.0,0.0],\
            [(self.AGE_MAX_POS[3][0]+self.AGE_MAX_POS[3][1])/2.0,0.0]]

        # Create spaces
        self.action_space = []
        self.observation_space = []
        for i in range(self.AGE_NUM):
            high_a = np.array([self.MAX_VEL, self.MAX_VEL])
            high_s = np.array([self.AGE_MAX_POS[i][1], self.MAX_POS_Y, self.MAX_POS_X, self.MAX_POS_Y, self.MAX_BALL_VEL, self.MAX_BALL_VEL])
            low_s = np.array([self.AGE_MAX_POS[i][0], -self.MAX_POS_Y, -self.MAX_POS_X, -self.MAX_POS_Y, -self.MAX_BALL_VEL, -self.MAX_BALL_VEL])
            self.action_space.append(spaces.Box(-high_a, high_a))
            self.observation_space.append(spaces.Box(low_s, high_s))

        # Initialize
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = []
        vb = self.np_random.uniform(low=0.0, high=self.MAX_BALL_VEL/10.0, size=(2))
        pb = self.np_random.uniform(low=-self.MAX_POS_Y, high=self.MAX_POS_Y, size=(1))
        vb = np.array([0.0, 0.0])
        for i in range(self.AGE_NUM):
            self.state.append([self.AGE_INI_POS[i][0], self.AGE_INI_POS[i][1], 0.0, pb, vb[0], vb[1]])
        self.state = np.array(self.state)
        return self._get_obs()

    def step(self, action):
        s = self.state
        a = np.array([ [np.clip(action[i][0], -self.MAX_VEL, self.MAX_VEL), np.clip(action[i][1], -self.MAX_VEL, self.MAX_VEL)] for i in range(self.AGE_NUM) ])

        # update the position of each agent
        ns = self._dynamics(s, a, self.DT)
        for i in range(self.AGE_NUM):
            ns[i,0] = np.clip(ns[i,0], self.AGE_MAX_POS[i][0]+self.AGE_SIZE[i][0]/2.0, self.AGE_MAX_POS[i][1]-self.AGE_SIZE[i][0]/2.0)
            ns[i,1] = np.clip(ns[i,1], -self.MAX_POS_Y+self.AGE_SIZE[i][1]/2.0, self.MAX_POS_Y-self.AGE_SIZE[i][1]/2.0)
            ns[i,2] = np.clip(ns[i,2], -self.MAX_POS_X, self.MAX_POS_X)
            ns[i,3] = np.clip(ns[i,3], -self.MAX_POS_Y, self.MAX_POS_Y)
            ns[i,4] = np.clip(ns[i,4], -self.MAX_BALL_VEL, self.MAX_BALL_VEL)
            ns[i,5] = np.clip(ns[i,5], -self.MAX_BALL_VEL, self.MAX_BALL_VEL)
            self.state[i] = ns[i]

        # reward design
        reward = np.zeros(self.AGE_NUM)
        if np.abs(ns[0,2]) > 0.9 * self.MAX_POS_X:
            done = True
        else:
            done = False
        for i in range(self.AGE_NUM):
            reward[i] = self._get_reward(self.AGE_TASK[i], self.AGE_GAIN[i], self.state[i], a[i])

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        s = self.state
        rtv = []
        for i in range(self.AGE_NUM):
            if self.AGE_TASK[i] == 0:
                rtv.append( np.array(s[i]))
            else:
                rtv.append( np.array(s[i]))
        return rtv

    def _dynamics(self, s, a, dt):
        eb = self.OBJ_RESTITUTION
        ma = 1.0
        mb = 0.01
        ns = s
        pb = s[0,2:4]
        vb = s[0,4:6]
        for i in range(self.AGE_NUM):
            vx = a[i,0]
            vy = a[i,1]
            ns[i,0:2] += np.array([vx, vy]) * dt
        for i in range(self.AGE_NUM):
            flag = False
            l = np.linalg.norm(s[i,0:2]-pb)
            tb = np.arctan2(pb[1]-ns[i,1],pb[0]-ns[i,0])
            if tb > np.pi/2.0:
                tb = np.pi - tb
            elif tb < -np.pi/2.0:
                tb = -np.pi - tb
            if np.absolute(tb) < np.arctan2(self.AGE_SIZE[i][1],self.AGE_SIZE[i][0]):
                flag = True
                l_ab = self.OBJ_SIZE + self.AGE_SIZE[i][0]/2.0/np.abs(np.cos(tb))
                is_coll = l <= l_ab
            else:
                l_ab = self.OBJ_SIZE + self.AGE_SIZE[i][1]/2.0/np.abs(np.sin(tb))
                is_coll = l <= l_ab
            tb = np.arctan2(pb[1]-ns[i,1],pb[0]-ns[i,0])
            if is_coll:
                tb = np.arctan2(pb[1]-ns[i,1],pb[0]-ns[i,0])
                va = np.array([a[i,0], a[i,1]])
                vb = ( (1.0+eb)*ma*va + (mb-ma*eb)*vb ) / (ma + mb)
                pb[0] += np.abs(l_ab-l)*np.cos(tb) + vb[0] * dt
                pb[1] += np.abs(l_ab-l)*np.sin(tb) + vb[1] * dt
            else:
                pb += vb * dt
                vb = vb
            if np.abs(pb[0])>self.MAX_POS_X:
                pb[0] = np.sign(pb[0])*self.MAX_POS_X+(np.sign(pb[0])*self.MAX_POS_X-pb[0])
                vb[0] = -vb[0]*eb*0.9
            if np.abs(pb[1])>self.MAX_POS_Y:
                pb[1] = np.sign(pb[1])*self.MAX_POS_Y+(np.sign(pb[1])*self.MAX_POS_Y-pb[1])
                vb[1] = -vb[1]*eb
        for i in range(self.AGE_NUM):
            ns[i,2:4] = pb
            ns[i,4:6] = vb

        return ns

    def _get_reward(self, num, g, s, a):
        rtv = 0.0
        if num < 2:
            rtv = s[2] / self.MAX_POS_X
        else:
            rtv = - s[2] / self.MAX_POS_X
        return g * rtv

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)


    def render(self, mode='human'):
        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(200*self.MAX_POS_X/self.MAX_POS_Y),200)
            self.viewer.set_bounds(-self.MAX_POS_X,self.MAX_POS_X,-self.MAX_POS_Y,self.MAX_POS_Y)
        self.viewer.draw_line((-self.MAX_POS_X/2.0, -self.MAX_POS_Y), (-self.MAX_POS_X/2.0, self.MAX_POS_Y))
        self.viewer.draw_line((0.0, -self.MAX_POS_Y), (0.0, self.MAX_POS_Y))
        self.viewer.draw_line((self.MAX_POS_X/2.0, -self.MAX_POS_Y), (self.MAX_POS_X/2.0, self.MAX_POS_Y))
        for i in range(self.AGE_NUM):
            ages = self.viewer.draw_polygon([(-self.AGE_SIZE[i][0]/2.0,-self.AGE_SIZE[i][1]/2.0), (-self.AGE_SIZE[i][0]/2.0,self.AGE_SIZE[i][1]/2.0),
            (self.AGE_SIZE[i][0]/2.0,self.AGE_SIZE[i][1]/2.0), (self.AGE_SIZE[i][0]/2.0,-self.AGE_SIZE[i][1]/2.0)])
            if i==0:
                ages.set_color(22.0/255.0, 108.0/255.0, 156.0/255.0)
            elif i==1:
                ages.set_color(64.0/255.0, 136.0/255.0, 108.0/255.0)
            elif i==2:
                ages.set_color(174.0/255.0, 101.0/255.0, 46.0/255.0)
            elif i==3:
                ages.set_color(185.0/255.0, 134.0/255.0, 164.0/255.0)
            jtransform = rendering.Transform(rotation=0.0, translation=[s[i,0],s[i,1]])
            ages.add_attr(jtransform)
        circ = self.viewer.draw_circle(self.OBJ_SIZE)
        circ.set_color(0.8, 0.0, 0.0)
        circ.add_attr(rendering.Transform(rotation=0.0, translation=(s[0,2], s[0,3])))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
