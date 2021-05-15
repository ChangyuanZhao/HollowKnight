<<<<<<< HEAD
import torch
import utilize.Helper
import time
import collections

from utilize.Actions import take_action, restart, take_direction
from ReplayMemory import ReplayMemory
from Model import actNet, moveNet
from utilize.GetHP import Hp_getter
from DQN import DQN
from Agent import Agent
from utilize.FrameBuffer import FrameBuffer
from utilize.Helper import mean
from RunEpisode import run_episode


Actions = ['Attack', 'Attack_Up',
           'Short_Jump', 'Mid_Jump', 'Skill_Up',
           'Skill_Down', 'Rush', 'Cure']
Directions = ['Move_Left', 'Move_Right', 'Turn_Left', 'Turn_Right']

class config:
    def __init__(self):
        self.device = torch.device("cuda")
        self.DELEY_REWARD = 1
        self.WIDTH = 400
        self.HEIGHT = 200
        self.FRAMEBUFFERSIZE = 4
        self.MEMORY_SIZE = 200
        self.actdim = 7
        self.movedim = 4
        self.MEMORY_WARMUP_SIZE = 24 # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
        self.BATCH_SIZE = 10 # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
        self.save_every = 100
        self.GAMMA = 0.95
        self.INPUT_SHAPE = (self.FRAMEBUFFERSIZE, self.HEIGHT, self.WIDTH, 3)
        self.LEARNING_RATE = 0.01


if __name__ == '__main__':
    opt = config()

#thread1 = utilize.FrameBuffer.FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
#thread1.start()
#while (len(thread1.buffer) < FRAMEBUFFERSIZE):
    #time.sleep(0.1)

#stations = thread1.get_buffer()
#stations = torch.Tensor(stations)


    actmodel = actNet(opt).to(opt.device)
    movemodel = moveNet(opt).to(opt.device)

    algorithm = DQN(actmodel, movemodel, opt, gamma=opt.GAMMA, learning_rate=opt.LEARNING_RATE)
    agent = Agent(opt.actdim, algorithm, opt, e_greed=0.12, e_greed_decrement=1e-6)

    act_rmp_correct = ReplayMemory(opt.MEMORY_SIZE, file_name='./act_memory')  # experience pool
    move_rmp_correct = ReplayMemory(opt.MEMORY_SIZE, file_name='./move_memory')  # experience pool
    paused = True
    paused = utilize.Helper.pause_game(paused, actmodel, movemodel)
    PASS_COUNT = 0
    hp = Hp_getter()
    total_remind_hp = 0

    max_episode = 30000
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1
        # if episode % 20 == 1:
        #     algorithm.replace_target()

        total_reward, total_step, PASS_COUNT, remind_hp, boss_hp = run_episode(hp, algorithm, agent, act_rmp_correct,
                                                                    move_rmp_correct, PASS_COUNT, opt, paused)
        total_remind_hp += remind_hp
        torch.cuda.empty_cache()
        paused = False
        paused = utilize.Helper.pause_game(paused, actmodel, movemodel, episode)
        if episode % opt.save_every == 0:
            torch.save(actmodel.state_dict(), 'model/act_%s' % episode)
            torch.save(movemodel.state_dict(), 'model/move_%s' % episode)
        print("Episode: ", episode, ", pass_count: ", PASS_COUNT, ", boss hp", boss_hp, ", hp:", remind_hp)

=======
import torch
import utilize.Helper
import time
import collections

from utilize.Actions import take_action, restart, take_direction
from ReplayMemory import ReplayMemory
from Model import actNet, moveNet
from utilize.GetHP import Hp_getter
from DQN import DQN
from Agent import Agent
from utilize.FrameBuffer import FrameBuffer
from utilize.Helper import mean
from RunEpisode import run_episode


Actions = ['Attack', 'Attack_Up',
           'Short_Jump', 'Mid_Jump', 'Skill_Up',
           'Skill_Down', 'Rush', 'Cure']
Directions = ['Move_Left', 'Move_Right', 'Turn_Left', 'Turn_Right']

class config:
    def __init__(self):
        self.device = torch.device("cuda")
        self.DELEY_REWARD = 1
        self.WIDTH = 400
        self.HEIGHT = 200
        self.FRAMEBUFFERSIZE = 4
        self.MEMORY_SIZE = 200
        self.actdim = 7
        self.movedim = 4
        self.MEMORY_WARMUP_SIZE = 24 # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
        self.BATCH_SIZE = 10 # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
        self.save_every = 100
        self.GAMMA = 0.95
        self.INPUT_SHAPE = (self.FRAMEBUFFERSIZE, self.HEIGHT, self.WIDTH, 3)
        self.LEARNING_RATE = 0.01


if __name__ == '__main__':
    opt = config()

#thread1 = utilize.FrameBuffer.FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
#thread1.start()
#while (len(thread1.buffer) < FRAMEBUFFERSIZE):
    #time.sleep(0.1)

#stations = thread1.get_buffer()
#stations = torch.Tensor(stations)


    actmodel = actNet(opt).to(opt.device)
    movemodel = moveNet(opt).to(opt.device)

    algorithm = DQN(actmodel, movemodel, opt, gamma=opt.GAMMA, learning_rate=opt.LEARNING_RATE)
    agent = Agent(opt.actdim, algorithm, opt, e_greed=0.12, e_greed_decrement=1e-6)

    act_rmp_correct = ReplayMemory(opt.MEMORY_SIZE, file_name='./act_memory')  # experience pool
    move_rmp_correct = ReplayMemory(opt.MEMORY_SIZE, file_name='./move_memory')  # experience pool
    paused = True
    paused = utilize.Helper.pause_game(paused, actmodel, movemodel)
    PASS_COUNT = 0
    hp = Hp_getter()
    total_remind_hp = 0

    max_episode = 30000
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1
        # if episode % 20 == 1:
        #     algorithm.replace_target()

        total_reward, total_step, PASS_COUNT, remind_hp, boss_hp = run_episode(hp, algorithm, agent, act_rmp_correct,
                                                                    move_rmp_correct, PASS_COUNT, opt, paused)
        total_remind_hp += remind_hp
        torch.cuda.empty_cache()
        paused = False
        paused = utilize.Helper.pause_game(paused, actmodel, movemodel, episode)
        if episode % opt.save_every == 0:
            torch.save(actmodel.state_dict(), 'model/act_%s' % episode)
            torch.save(movemodel.state_dict(), 'model/move_%s' % episode)
        print("Episode: ", episode, ", pass_count: ", PASS_COUNT, ", boss hp", boss_hp, ", hp:", remind_hp)

>>>>>>> d00f7fee553731b8aa5db54bf247e6437c5d6052
