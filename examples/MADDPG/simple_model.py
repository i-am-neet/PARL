#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

LAST_IMAGE_SIZE = 81

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 critic_in_dim,
                 continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        # personal_dim is contain p_vel, p_pos, goal_pos, lidars, next_dir, and A*'s route
        # equation is : world_dim + world_dim + world_dim + amount of scan + world_dim + route's size * world_dim
        self.personal_dim = 8
        # collaborate_dim is contain other_gridmap() & other_plan_gridmap()
        # gridmap is flatten of (grid size * grid size)
        # where grid size is default 9
        # Or using -1 to auto-detect
        self.plan_dim = 16
        self.lidar_dim = 12
        self.collaborate_dim_1 = LAST_IMAGE_SIZE + LAST_IMAGE_SIZE
        assert (self.personal_dim + self.plan_dim + self.lidar_dim + self.collaborate_dim_1) == obs_dim
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.personal_model = nn.Sequential(
            nn.Linear(self.personal_dim, 32, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(32, 32, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(32, 8, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
        )
        self.plan_model = nn.Sequential(
            nn.Linear(self.plan_dim, 64, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(64, 64, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(64, 16, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
        )
        self.lidar_model = nn.Sequential(
            nn.Linear(self.lidar_dim, 64, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(64, 64, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(64, 16, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
        )
        self.collaborate_model_1 = nn.Sequential(
            nn.Linear(self.collaborate_dim_1, 256, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(256, 256, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(256, 100, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
        )
        self.mixed_model = nn.Sequential(
            nn.Linear(8 + 16 + 16 + 100, 256, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(256, 128, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.LeakyReLU(),
            nn.Linear(128, act_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
        )

    def forward(self, obs):
        i1, i2, i3, i4 = paddle.split(obs, [self.personal_dim, self.plan_dim, self.lidar_dim, self.collaborate_dim_1], axis=-1)
        o1 = self.personal_model(i1)
        o2 = self.plan_model(i2)
        o3 = self.lidar_model(i3)
        o4 = self.collaborate_model_1(i4)
        means = self.mixed_model(paddle.concat([o1, o2, o3, o4], axis=-1))
        if self.continuous_actions:
            act_std = self.std_fc(hid5)
            return (means, act_std)
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1_size = 512
        hid2_size = 512
        out_dim = 1
        self.fc1 = nn.Linear(
            critic_in_dim,
            hid1_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(
            hid2_size,
            out_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        inputs = paddle.concat(obs_n + act_n, axis=1)
        hid1 = F.leaky_relu(self.fc1(inputs))
        hid2 = F.leaky_relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        Q = paddle.squeeze(Q, axis=1)
        return Q
