import os
import gym
import numpy as np
import parl
import random
import collections

from parl import layers
from parl.algorithms import DQN  # from parl.algorithms import PolicyGradient
from parl.utils import logger

import paddle.fluid as fluid

import ple
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w

from Model import Model
from Agent import Agent
from ReplayMemory import ReplayMemory



actions = {"up": K_w}
LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.0001 # 学习率


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0

    env.init()
    steps = 0
    action_set = env.getActionSet()
    while not env.game_over():
        steps += 1
        if (steps == 1):
            continue

        obs = list(env.getGameState().values())
        action_idx = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        act = action_set[action_idx]

        reward = env.act(act)
        done = env.game_over()
        next_obs = list(env.getGameState().values())

        rpm.append((obs, action_idx, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (steps % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
                batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                        batch_next_obs,
                                        batch_done)  # s,a,r,s',done

        total_reward += reward

    env.reset_game()
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    action_set = env.getActionSet()
    for i in range(5):
        env.init()
        episode_reward = 0
        steps = 0
        while not env.game_over():
            steps += 1
            if (steps == 1):
                continue

            obs = list(env.getGameState().values())
            action_idx = agent.predict(obs)  # 预测动作，只选最优动作

            # 神经网络输出转化为实际动作
            act = action_set[action_idx]
            reward = env.act(act)
            episode_reward += reward

            # 避免停不下来
            if (steps == 5000):
                break
        
        logger.info('[Test] episode:{}, steps:{}, reward:{}, score:{}'.format(
            i, steps, episode_reward, env.score()))
        env.reset_game()
        eval_reward.append(episode_reward)
    return np.mean(eval_reward), steps



def train():
    # 创建环境
    game = FlappyBird()
    env_1 = PLE(game, fps=30, display_screen=False)
    env_2 = PLE(game, fps=30, display_screen=True)
    obs_dim = len(env_1.getGameState())
    act_dim = len(env_1.getActionSet())
    print('action set:', env_1.getActionSet())
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    algorithm = DQN(model, act_dim = act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim = obs_dim,
        act_dim = act_dim,
        e_greed = 0.3,
        e_greed_decrement = 1e-6
    )

    # 加载模型
    save_path = './flappybird.ckpt'
    if os.path.exists(save_path):
        agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env_1, agent, rpm)

    max_episode = 2000

    # 开始训练
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train
        for i in range(0, 100):
            total_reward, steps = run_episode(env_1, agent, rpm)
            episode += 1

        # test
        eval_reward, steps = evaluate(env_2, agent)
        logger.info('[episode:{}], e_greed:{:.6f}, steps:{}, test_reward:{}'.format(
            episode, agent.e_greed, steps, eval_reward))
        # 保存模型
        ckpt = './models/episode_{}.ckpt'.format(episode)
        agent.save(ckpt)

    # 训练结束，保存模型
    save_path = './flappybird.ckpt'
    agent.save(save_path)

import pygame
def drawText(screen,text,posx=0,posy=0,textHeight=48,fontColor=(0,0,0),backgroudColor=(255,255,255)):
    fontObj = pygame.font.Font(None, textHeight)  # 通过字体文件获得字体对象
    textSurfaceObj = fontObj.render(text, True,fontColor,backgroudColor)  # 配置要显示的文字
    textRectObj = textSurfaceObj.get_rect()  # 获得要显示的对象的rect
    textRectObj.right, textRectObj.top = posx, posy  # 设置显示对象的坐标
    screen.blit(textSurfaceObj, textRectObj)  # 绘制字
    pygame.display.update()

def test():
    # 创建环境
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True)
    obs_dim = len(env.getGameState())
    act_dim = len(env.getActionSet())
    print('action set:', env.getActionSet())
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    algorithm = DQN(model, act_dim = act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim = obs_dim,
        act_dim = act_dim,
        e_greed = 0.3,
        e_greed_decrement = 1e-6
    )

    # 加载模型
    save_path = './DQN/checkpoints/episode_V14600.ckpt'
    print('checkpoints:', save_path)
    if os.path.exists(save_path):
        logger.info('load ckpt success!')
        agent.restore(save_path)
    else:
        logger.error('load ckpt error!')
        
    action_set = env.getActionSet()
    env.init()
    episode_reward = 0
    steps = 0
    while not env.game_over():
        steps += 1
        if (steps == 1):
            continue
        obs = list(env.getGameState().values())
        action_idx = agent.predict(obs)  # 预测动作，只选最优动作
        act = action_set[action_idx]
        reward = env.act(act)
        episode_reward += reward
        reward_str = str(int(episode_reward))
        drawText(env.game.screen, reward_str, 288, 0, 48, (255,0,0), (255,255,255))
    env.reset_game()
    logger.info('[Test] steps:{}, reward:{}'.format(
        steps, episode_reward))


if __name__ == "__main__":
    # train()
    test()
