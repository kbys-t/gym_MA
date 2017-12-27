# coding:utf-8
"""
Move to goal with formation
Goal moves along with sine curve
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MoveFormSinEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.DT = 0.02
        # Physical params
        # Limitation
        self.MAX_VEL = 1.0
        self.MAX_ANG_VEL = 2.0 * np.pi
        self.MAX_POS = 1.0
        # reference
        self.TARGET_FORM = np.array([np.pi/3.0, 0.2])
        self.TARGET_FORM[1] *= self.MAX_POS / np.sin(self.TARGET_FORM[0])
        self.TARGET_FORM = np.append(self.TARGET_FORM, 0.5 * np.sin(self.TARGET_FORM[0]) * self.TARGET_FORM[1]**2)
        self.TARGET_WAVE = np.array([10.0, 0.5])

        # for multi agents
        self.AGE_NUM = 3
        self.AGE_TASK = [0, 1, 2]
        self.AGE_GAIN = [1.0, 1.0, 1.0]

        # Create spaces
        self.action_space = []
        self.observation_space = []
        for i in range(self.AGE_NUM):
            if self.AGE_TASK[i] == 0:
                high_a = np.array([self.MAX_VEL, self.MAX_ANG_VEL])
                high_s = np.array([self.MAX_POS, self.MAX_POS, 1.0, 1.0])
                low_s = - np.array([self.MAX_POS, self.MAX_POS, 1.0, 1.0])
                # high_s = np.array([self.MAX_POS, self.MAX_POS, 1.0, 1.0, 2.0*self.MAX_POS, 2.0*self.MAX_POS])
                # low_s = - np.array([self.MAX_POS, self.MAX_POS, 1.0, 1.0, 2.0*self.MAX_POS, 2.0*self.MAX_POS])
                self.action_space.append(spaces.Box(-high_a, high_a))
                self.observation_space.append(spaces.Box(low_s, high_s))
            else:
                high_a = np.array([self.MAX_VEL, self.MAX_ANG_VEL])
                high_s = np.array([np.sqrt(2.0)*self.MAX_POS, 1.0, 1.0, np.sqrt(2.0)*self.MAX_POS, 1.0, 1.0])
                low_s = - np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
                self.action_space.append(spaces.Box(-high_a, high_a))
                self.observation_space.append(spaces.Box(low_s, high_s))

        # Initialize
        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.AGE_NUM, 7))
        self.state[:, 0] -= self.MAX_POS
        self.target = self.np_random.uniform(low=-0.25, high=0.25, size=(2, 1))
        self.target[0] -= self.MAX_POS*2.0  # force to edge
        for i in range(self.AGE_NUM):
            self.state[i, 3:] = self._measure(i)
        return self._get_obs()

    def _step(self, action):
        s = self.state
        a = np.array([ [np.clip(action[i][0], -self.MAX_VEL, self.MAX_VEL), np.clip(action[i][1], -self.MAX_ANG_VEL, self.MAX_ANG_VEL)] for i in range(self.AGE_NUM) ])
        # update the position of each agent
        for i in range(self.AGE_NUM):
            ns = self._dynamics(s[i, 0:3], a[i], self.DT)
            ns[0] = np.clip(ns[0], -self.MAX_POS, self.MAX_POS)
            ns[1] = np.clip(ns[1], -self.MAX_POS, self.MAX_POS)
            ns[2] = angle_normalize(ns[2])
            self.state[i, 0:3] = ns

        # update the target position
        ysign = 1.0 if np.absolute(self.target[0] / self.MAX_POS) < 0.5 else -1.0
        self.target[1] = self.target[1] * np.cos(2.0*np.pi / self.TARGET_WAVE[0]*self.DT) + ysign * np.sqrt(self.TARGET_WAVE[1]**2 - self.target[1]**2) * np.sin(2.0*np.pi / self.TARGET_WAVE[0]*self.DT)
        self.target[0] += 2.0 * self.MAX_POS / self.TARGET_WAVE[0]*self.DT
        self.target = np.clip(self.target, -self.MAX_POS, self.MAX_POS)

        # measure the other states
        for i in range(self.AGE_NUM):
            self.state[i, 3:] = self._measure(i)

        # reward design
        reward = np.zeros(self.AGE_NUM)
        done = False
        for i in range(self.AGE_NUM):
            reward[i] = self._get_reward(self.AGE_TASK[i], self.AGE_GAIN[i], self.state[i], a[i])

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        rtv = []
        for i in range(self.AGE_NUM):
            if self.AGE_TASK[i] == 0:
                rtv.append( np.array([ s[i,3], s[i,4], np.cos(s[i,2]), np.sin(s[i,2]) ]) )
                # rtv.append( np.array([ s[i,0], s[i,1], np.cos(s[i,2]), np.sin(s[i,2]), s[i,3], s[i,4] ]) )
            else:
                rtv.append( np.array([ s[i,3], np.cos(s[i,4]), np.sin(s[i,4]), s[i,5], np.cos(s[i,6]), np.sin(s[i,6]) ]) )
        return rtv

    def _dynamics(self, s, a, dt):
        # http://myenigma.hatenablog.com/entry/20140301/1393648106
        theta = s[2]
        v = a[0]
        w = a[1]
        dtheta = w*dt
        dx = v / w * ( np.sin(theta+dtheta) - np.sin(theta) ) if np.absolute(w) > 1e-12 else v * np.cos(theta)
        dy = - v / w * ( np.cos(theta+dtheta) - np.cos(theta) ) if np.absolute(w) > 1e-12 else v * np.sin(theta)
        return s + np.array([dx, dy, dtheta])

    def _measure(self, num):
        rtv = np.zeros((4, 1))
        if self.AGE_TASK[num] == 0:
            # relative position of target
            dp = self.target - self.state[num, 0:2].reshape((-1, 1))
            c = np.cos(self.state[num, 2])
            s = np.sin(self.state[num, 2])
            rtv[0:2] = np.array([[c, s], [-s, c]]).dot(dp)
        else:
            # distances and angles of respective agents
            addnum = 0
            for i in range(self.AGE_NUM):
                if i != num:
                    rtv[addnum*2] = np.linalg.norm(self.state[i, 0:2] - self.state[num, 0:2])
                    rtv[addnum*2+1] = angle_normalize( np.arctan2(self.state[i, 1]-self.state[num, 1], self.state[i, 0]-self.state[num, 0]) - self.state[num, 2] )
                    addnum += 1
        return rtv.reshape((-1,))

    def _get_reward(self, num, g, s, a):
        rtv = 0.0
        if num == 0:
            # target tracking
            rtv = -1.0 + 2.0*np.exp( -1.0 * np.linalg.norm(s[3:5]) )
        elif num == 1:
            # triangle formation
            dth = s[4] - s[6] if s[4] > s[6] else s[6] - s[4]
            dth = 2.0*np.pi - dth if dth > np.pi else dth
            area = 0.5 * np.sin(dth) * s[3] * s[5]
            rtv = -1.0 + 2.0*np.exp( -1.0 * ( \
                + np.absolute(dth - self.TARGET_FORM[0]) \
                + np.absolute(area - self.TARGET_FORM[2]) \
                + np.absolute(s[3] - s[5]) \
                ) )
        elif num == 2:
            # energy minimization
            rtv = 1.0 - ( np.absolute(a[0]) / self.MAX_VEL + np.absolute(a[1]) / self.MAX_ANG_VEL )
        return g * rtv

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.MAX_POS,self.MAX_POS,-self.MAX_POS,self.MAX_POS)


        self.viewer.draw_line((-self.MAX_POS, 0), (self.MAX_POS, 0))
        self.viewer.draw_line((0, -self.MAX_POS), (0, self.MAX_POS))
        for i in range(self.AGE_NUM):
            r = 0.05

            jtransform = rendering.Transform(rotation=0.0, translation=[self.target[0],self.target[1]])
            circ = self.viewer.draw_circle(0.02)
            circ.set_color(0.2, 0.2, 0.2)
            circ.add_attr(jtransform)

            ages = self.viewer.draw_polygon([(-r,-r/1.5), (-r,r/1.5), (r,0), (r,0)])
            if i==0:
                ages.set_color(0.4, 0.761, 0.647)
            elif i==1:
                ages.set_color(0.988, 0.553, 0.384)
            else:
                ages.set_color(0.553, 0.627, 0.796)
            jtransform = rendering.Transform(rotation=s[i,2], translation=[s[i,0],s[i,1]])
            ages.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
