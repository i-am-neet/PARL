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

import os
import time
import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
# TODO multiagent_simple_env will be deprecated
# from parl.env.multiagent_env import MAenv
from parl.env.multiagent_simple_env import MAenv
from parl.utils import logger, summary
import cv2
from gym import spaces
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logging_counter = 0
total_cost_time_n = 0
total_collide_times_n = 0

CRITIC_LR = 0.01 #1e-4 #0.01  # learning rate for the critic model
ACTOR_LR = 0.01 #1e-4 #0.01  # learning rate of the actor model
GAMMA = 0.95 #0.9 #0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_EPISODES = 60e+3 #25000  # stop condition:number of episodes
MAX_STEP_PER_EPISODE = 250#100 #25  # maximum step per episode
STAT_RATE = 200 # 1000 # statistical interval of save model or count reward

LAST_DATA_SIZE = 81

def near_has_agent(agent, obs_n):
    pa = obs_n[agent.agent_index][2:4]
    for i in range(len(obs_n)):
        pb = obs_n[i][2:4]
        if i == agent.agent_index: continue
        if np.linalg.norm(pa - pb) < 0.22:
            return True
    return False

def advice_control(obs_n, action_n):
    for i in range(len(obs_n)):
        for j in range(i+1, len(obs_n)):
            pa = obs_n[i][2:4]
            pb = obs_n[j][2:4]
            if np.linalg.norm(pa - pb) < 0.1:
                a_action = np.zeros(9)
                b_action = np.zeros(9)
                a_u_index = np.argmax(action_n[i])
                b_u_index = np.argmax(action_n[j])
                a_c_index = 0
                b_c_index = 0
                if b_u_index == 0:
                    i1 = (a_u_index - 2)%8 if (a_u_index - 2)%8 !=0 else 8 # right way
                    i2 = (a_u_index + 2)%8 if (a_u_index + 2)%8 !=0 else 8 # left way
                    i3 = (a_u_index + 4)%8 if (a_u_index + 4)%8 !=0 else 8 # inverse way
                    a_c_index = np.random.choice([0, i1, i2, i3])
                    a_action[a_c_index] = 1
                    action_n[i] = a_action
                else:
                    i1 = (b_u_index - 2)%8 if (b_u_index - 2)%8 !=0 else 8 # right way
                    i2 = (b_u_index + 2)%8 if (b_u_index + 2)%8 !=0 else 8 # left way
                    i3 = (b_u_index + 4)%8 if (b_u_index + 4)%8 !=0 else 8 # inverse way
                    b_c_index = np.random.choice([0, i1, i2, i3])
                    b_action[b_c_index] = 1
                    action_n[j] = b_action
    return action_n

def expert_control(expert_a, agent_a):
    e_u_index = np.argmax(expert_a)
    a_u_index = np.argmax(agent_a)
    ctrl_action = np.zeros(9)
    ctrl_index = 0
    if a_u_index == 0: ctrl_index = 0 # stop
    if a_u_index == 1: ctrl_index = e_u_index # same way
    if a_u_index == 2: ctrl_index = (e_u_index + 4)%8 if (e_u_index + 4)%8 !=0 else 8 # inverse way
    if a_u_index == 3: ctrl_index = (e_u_index - 2)%8 if (e_u_index - 2)%8 !=0 else 8 # right way
    if a_u_index == 4: ctrl_index = (e_u_index + 2)%8 if (e_u_index + 2)%8 !=0 else 8 # left way
    ctrl_action[ctrl_index] = 1
    return ctrl_action

def random_expert(expert_action_n):
    p_n = []
    for e in expert_action_n:
        p = np.zeros(9)
        p[np.argmax(e)] = 0.6
        p[(np.argmax(e)-1)%9] = 0.2
        p[(np.argmax(e)+1)%9] = 0.2
        p_n.append(p)
    random_index_n = [np.random.choice([0,1,2,3,4,5,6,7,8], p=p) for p in p_n]
    random_expert_action_n = []
    for ri in random_index_n:
        z = np.zeros(9)
        z[ri] = 1
        random_expert_action_n.append(z)

def run_episode(env, agents):
    obs_n = env.reset(testing=True) if args.restore and args.show else env.reset()
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    last_image_n = [np.split(o, [-LAST_DATA_SIZE])[-1] for o in obs_n]
    obs_n = [np.concatenate((o, i)) for o, i in zip(obs_n, last_image_n)]
    while True:
        steps += 1
        expert_action_n = env.get_expert_action_n()
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        if False: # test expert
            ctrl_action_n = [expert_control(e_a, a_a) for e_a, a_a in zip(expert_action_n, action_n)]
            next_obs_n, reward_n, done_n, info_n = env.step(ctrl_action_n)
        elif False: # advice
            advice_control_n = advice_control(obs_n, action_n)
            next_obs_n, reward_n, done_n, info_n = env.step(advice_control_n)
        elif False: # advice w/ expert
            advice_control_n = advice_control(obs_n, expert_action_n)
            next_obs_n, reward_n, done_n, info_n = env.step(advice_control_n)
        elif True: # switch by check nearest
            for i in range(env.n):
                action_n[i] = action_n[i] if near_has_agent(agents[i], obs_n) else expert_action_n[i]
            next_obs_n, reward_n, done_n, info_n = env.step(action_n)
        else:
            # Normal
            next_obs_n, reward_n, done_n, info_n = env.step(action_n)
        last_image_n = [np.split(o, [-LAST_DATA_SIZE*2, -LAST_DATA_SIZE])[-2] for o in obs_n]
        next_obs_n = [np.concatenate((o, i)) for o, i in zip(next_obs_n, last_image_n)]
        # if any(info_n['n']):
        #     print(f"At step {steps}: {info_n['n']}")
        done = all(done_n)
        terminal = (steps >= MAX_STEP_PER_EPISODE)

        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # check the end of an episode
        if done or terminal:
            if args.restore:
                global total_cost_time_n, total_collide_times_n, logging_counter
                cost_time_n, collid_times_n = env.get_info()
                eval_logger.info(logging_counter)
                eval_logger.info(f"COST TIME")
                eval_logger.info(f"---------")
                eval_logger.info(f"{cost_time_n}\n")
                eval_logger.info(f"COLLISION TIMES")
                eval_logger.info(f"---------------")
                eval_logger.info(f"{collid_times_n}\n")
                total_cost_time_n = total_cost_time_n + np.array(cost_time_n)
                total_collide_times_n = total_collide_times_n + np.array(collid_times_n)
                logging_counter+=1
                if logging_counter%100 == 0:
                    eval_logger.info(f"mean cost time: {total_cost_time_n / 100}")
                    eval_logger.info(f"mean collide times: {total_collide_times_n / 100}")
                    total_cost_time_n = 0
                    total_collide_times_n = 0
            break

        # show animation
        if args.show:
            time.sleep(0.01)
            env.render()

        # get world's image
        # w = env.get_world_array()
        # cv2.imshow('My Image', cv2.cvtColor(w[0], cv2.COLOR_RGB2BGR))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # show model effect without training
        if args.restore and args.show:
            continue

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            if critic_loss != 0.0:
                summary.add_scalar('critic_loss/agent_%d' % i, critic_loss,
                                   agent.global_train_step)

    return total_reward, agents_reward, steps


def train_agent():
    env = MAenv(args.env, args.num)
# =======
#     env = MAenv(args.env, args.continuous_actions)
#     if args.continuous_actions:
#         assert isinstance(env.action_space[0], spaces.Box)

#     # print env info
# >>>>>>> 0b7030ac1a751e9bab5a1c903069e8a0124d18b3
    logger.info('agent num: {}'.format(env.n))
    logger.info('obs_shape_n: {}'.format(env.obs_shape_n))
    logger.info('act_shape_n: {}'.format(env.act_shape_n))
    logger.info('observation_space: {}'.format(env.observation_space))
    logger.info('action_space: {}'.format(env.action_space))
    for i in range(env.n):
        logger.info('agent {} obs_low:{} obs_high:{}'.format(
            i, env.observation_space[i].low, env.observation_space[i].high))
        logger.info('agent {} act_n:{}'.format(i, env.act_shape_n[i]))
        if (isinstance(env.action_space[i], spaces.Box)):
            logger.info('agent {} act_low:{} act_high:{} act_shape:{}'.format(
                i, env.action_space[i].low, env.action_space[i].high,
                env.action_space[i].shape))

    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
    logger.info('critic_in_dim: {}'.format(critic_in_dim))

    # build agents
    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim,
                        args.continuous_actions)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE)
        agents.append(agent)
    total_steps = 0
    total_episodes = 0

    episode_rewards = []  # sum of rewards for all agents
    agent_rewards = [[] for _ in range(env.n)]  # individual agent reward

    # if args.restore:
    if False:
        # restore model
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    # restore model & fine tune
    for i in range(len(agents)):
        model_file = './tuned_model/agent_' + str(i)
        print(f"fine tune: {model_file}")
        if not os.path.exists(model_file):
            raise Exception(
                'model file {} does not exits'.format(model_file))
        agents[i].restore(model_file)

    t_start = time.time()
    logger.info('Starting...')
    while total_episodes <= MAX_EPISODES:
        # run an episode
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
        if not args.restore:
            summary.add_scalar('train_reward/episode', ep_reward, total_episodes)
            summary.add_scalar('train_reward/step', ep_reward, total_steps)
        else:
            summary.add_scalar('restore/train_reward/episode', ep_reward, total_episodes)
            summary.add_scalar('restore/train_reward/step', ep_reward, total_steps)
        if args.show:
            print('episode {}, reward {}, agents rewards {}, steps {}'.format(
                total_episodes, ep_reward, ep_agent_rewards, steps))
        if total_episodes % 200 == 0:
            print('episode {}, reward {}, agents rewards {}, steps {}'.format(
                total_episodes, ep_reward, ep_agent_rewards, steps))

        # Record reward
        total_steps += steps
        total_episodes += 1
        episode_rewards.append(ep_reward)
        for i in range(env.n):
            agent_rewards[i].append(ep_agent_rewards[i])

        # Keep track of final episode reward
        if total_episodes % STAT_RATE == 0:
            mean_episode_reward = round(
                np.mean(episode_rewards[-STAT_RATE:]), 3)
            final_ep_ag_rewards = []  # agent rewards for training curve
            for rew in agent_rewards:
                final_ep_ag_rewards.append(round(np.mean(rew[-STAT_RATE:]), 2))
            use_time = round(time.time() - t_start, 3)
            logger.info(
                'Steps: {}, Episodes: {}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                .format(total_steps, total_episodes, mean_episode_reward,
                        final_ep_ag_rewards, use_time))
            t_start = time.time()
            if not args.restore:
                summary.add_scalar('mean_episode_reward/episode',
                                mean_episode_reward, total_episodes)
                summary.add_scalar('mean_episode_reward/step', mean_episode_reward,
                                total_steps)
                summary.add_scalar('use_time/1000episode', use_time,
                                total_episodes)
            else:
                summary.add_scalar('restore/mean_episode_reward/episode',
                                mean_episode_reward, total_episodes)
                summary.add_scalar('restore/mean_episode_reward/step', mean_episode_reward,
                                total_steps)
                summary.add_scalar('restore/use_time/1000episode', use_time,
                                total_episodes)

            # save model
            if not args.restore:
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i) + '_ep' + str(total_episodes)
                    agents[i].save(model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument(
        '--env',
        type=str,
        default='simple_spread_room',
        help='scenario of MultiAgentEnv')
    # auto save model, optional restore model
    parser.add_argument(
        '--show', action='store_true', default=False, help='display or not')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='restore or not, must have model_dir')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='directory for saving model')
    parser.add_argument(
        '--num',
        type=int,
        default=4,
        help='amount of agents & landmarks')
    parser.add_argument(
        '--continuous_actions',
        action='store_true',
        default=False,
        help='use continuous action mode or not')
    # parser.add_argument(
    #     '--max_episodes',
    #     type=int,
    #     default=25000,
    #     help='stop condition: number of episodes')

    args = parser.parse_args()
    print('========== args: ', args)
    if not args.restore:
        logger.set_dir(f'./train_log/{args.model_dir}/' + str(args.env))
    else:
        logger.set_dir(f'./train_log/restore/{args.model_dir}/' + str(args.env))

    from datetime import datetime
    eval_logger = setup_logger('eval_logger', f"./evals/{args.num}_agents_{datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M')}.log")

    # cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)

    train_agent()

    # cv2.destroyAllWindows()
