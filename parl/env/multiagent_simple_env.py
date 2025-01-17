#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from gym import spaces
    from multiagent.multi_discrete import MultiDiscrete
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    from parl.utils import logger

    class MAenv(MultiAgentEnv):
        """ multiagent environment warppers for maddpg
        """

        def __init__(self, scenario_name, amount):
            env_list = [
                'simple', 'simple_adversary', 'simple_crypto', 'simple_push',
                'simple_reference', 'simple_speaker_listener', 'simple_spread',
                'simple_tag', 'simple_world_comm', 'simple_spread_room'
            ]
            assert scenario_name in env_list, 'Env {} not found (valid envs include {})'.format(
                scenario_name, env_list)
            # load scenario from script
            scenario = scenarios.load(scenario_name + ".py").Scenario()
            # create world
            world = scenario.make_world(amount)
            # initial multiagent environment
            super().__init__(world, scenario.reset_world, scenario.reward,
                             scenario.observation, info_callback=scenario.info, done_callback=scenario.done, shared_viewer=True,
                             expert_callback=scenario.expert_action)
            self.obs_shape_n = [
                self.get_shape(self.observation_space[i])
                for i in range(self.n)
            ]
            self.act_shape_n = [
                self.get_shape(self.action_space[i]) for i in range(self.n)
            ]

        def get_shape(self, input_space):
            """
            Args:
                input_space: environment space

            Returns:
                space shape
            """
            if (isinstance(input_space, spaces.Box)):
                if (len(input_space.shape) == 1):
                    return input_space.shape[0]
                else:
                    return input_space.shape
            elif (isinstance(input_space, spaces.Discrete)):
                return input_space.n
            elif (isinstance(input_space, MultiDiscrete)):
                return sum(input_space.high - input_space.low + 1)
            else:
                print(
                    '[Error] shape is {}, not Box or Discrete or MultiDiscrete'
                    .format(input_space.shape))
                raise NotImplementedError

    logger.warning(
        'the `MAenv` from `parl.env.multiagent_simple_env` is deprecated since parl 2.0.4 and will be removed in parl 3.0. \n \
        We recomend you to use `from parl.env.multiagent_env import MAenv` instead, which supports continuous control.'
    )
except:
    raise ImportError(
        'Can not use MAenv from parl.env.multiagent_simple_env, \n \
        please pip install multiagent according to https://github.com/openai/multiagent-particle-envs \
        as well as `pip install gym==0.10.5`')
