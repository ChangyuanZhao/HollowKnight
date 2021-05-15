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

def run_episode(hp, algorithm, agent, act_rmp_correct, move_rmp_correct, PASS_COUNT, opt, pausedact_rmp_wrong=0,
                move_rmp_wrong=0):
    # 开始挑战
    restart()

    # 学习数据
   # for i in range(8):
       # if (len(move_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            # print("move learning")
         #   batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)
         #   algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        #if (len(act_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            # print("action learning")
           # batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(opt.BATCH_SIZE)
           # algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

    step = 0
    done = 0
    total_reward = 0

    start_time = time.time()

    # Deley Reward
    DeleyMoveReward = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyActReward = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyStation = collections.deque(maxlen=opt.DELEY_REWARD + 1)  # 1 more for next_station
    DeleyActions = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyDirection = collections.deque(maxlen=opt.DELEY_REWARD)

    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        if boss_hp_value > 800 and boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 9:
            break

    # 开始对战 获取帧图片
    thread1 = FrameBuffer(1, "FrameBuffer", opt.WIDTH, opt.HEIGHT, maxlen=opt.FRAMEBUFFERSIZE)
    thread1.start()

    last_hornet_y = 0
    while True:
        torch.cuda.empty_cache()
        step += 1
        #print(time.time())

        # 防止帧不够
        while len(thread1.buffer) < opt.FRAMEBUFFERSIZE:
            time.sleep(0.1)

        stations = thread1.get_buffer()
        #stations.to(opt.device)
        #print(stations.is_cuda)
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        player_x, player_y = hp.get_play_location()
        hornet_x, hornet_y = hp.get_hornet_location()
        soul = hp.get_souls()

        # 判定大黄蜂技能1
        hornet_skill1 = False
        if last_hornet_y > 32 and last_hornet_y < 32.5 and hornet_y > 32 and hornet_y < 32.5:
            hornet_skill1 = True
        #hornet_skill1 = False
        # 学习并输出
        move, action = agent.sample(stations, soul, hornet_x, hornet_y, player_x, hornet_skill1)

        take_direction(move)
        #print("Move"+Directions[move])
        take_action(action)
        #print("Action" + Actions[action])

        next_station = thread1.get_buffer()
        #next_station = next_station.to(opt.device)
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()


        # get reward
        move_reward = utilize.Helper.move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x,
                                             move, hornet_skill1)
        # print(move_reward)
        act_reward, done = utilize.Helper.action_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp,
                                                    next_player_x, next_hornet_x, next_hornet_x, action, hornet_skill1)
        # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)

        DeleyMoveReward.append(move_reward)
        DeleyActReward.append(act_reward)
        DeleyStation.append(stations)
        DeleyActions.append(action)
        DeleyDirection.append(move)
        DeleyStation.append(next_station)

        if len(DeleyStation) >= opt.DELEY_REWARD + 1:
            if DeleyMoveReward[0] != 0:
                move_rmp_correct.append((DeleyStation[0],DeleyDirection[0],DeleyMoveReward[0],DeleyStation[1],done))
            # if DeleyMoveReward[0] <= 0:
            #     move_rmp_wrong.append((DeleyStation[0],DeleyDirection[0],DeleyMoveReward[0],DeleyStation[1],done))

        if len(DeleyStation) >= opt.DELEY_REWARD + 1:
            if DeleyActReward[0] != 0:
                act_rmp_correct.append((DeleyStation[0],DeleyActions[0],DeleyActReward[0],DeleyStation[1],done))
            # if mean(DeleyActReward) <= 0:
            #     act_rmp_wrong.append((DeleyStation[0],DeleyActions[0],mean(DeleyActReward),DeleyStation[1],done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value

        total_reward += act_reward
        #paused = False
        #paused = utilize.Helper.pause_game(paused)
        # done 1 自己死了
        if done == 1:
            utilize.Actions.Nothing()
            break
        # done 2 BOSS死了
        elif done == 2:
            PASS_COUNT += 1
            utilize.Actions.Nothing()
            time.sleep(3)
            break

    thread1.stop()
    #batch_station, batch_actions, batch_reward, batch_next_station, batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)

    # 死亡后学习单次对战

    for i in range(8):
        if (len(move_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)
            algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done,opt.BATCH_SIZE)

        if (len(act_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            print("action learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(opt.BATCH_SIZE)
            algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done,opt.BATCH_SIZE)


    return total_reward, step, PASS_COUNT, self_hp, boss_hp_value


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

def run_episode(hp, algorithm, agent, act_rmp_correct, move_rmp_correct, PASS_COUNT, opt, pausedact_rmp_wrong=0,
                move_rmp_wrong=0):
    # 开始挑战
    restart()

    # 学习数据
   # for i in range(8):
       # if (len(move_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            # print("move learning")
         #   batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)
         #   algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        #if (len(act_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            # print("action learning")
           # batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(opt.BATCH_SIZE)
           # algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

    step = 0
    done = 0
    total_reward = 0

    start_time = time.time()

    # Deley Reward
    DeleyMoveReward = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyActReward = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyStation = collections.deque(maxlen=opt.DELEY_REWARD + 1)  # 1 more for next_station
    DeleyActions = collections.deque(maxlen=opt.DELEY_REWARD)
    DeleyDirection = collections.deque(maxlen=opt.DELEY_REWARD)

    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        if boss_hp_value > 800 and boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 9:
            break

    # 开始对战 获取帧图片
    thread1 = FrameBuffer(1, "FrameBuffer", opt.WIDTH, opt.HEIGHT, maxlen=opt.FRAMEBUFFERSIZE)
    thread1.start()

    last_hornet_y = 0
    while True:
        torch.cuda.empty_cache()
        step += 1
        #print(time.time())

        # 防止帧不够
        while len(thread1.buffer) < opt.FRAMEBUFFERSIZE:
            time.sleep(0.1)

        stations = thread1.get_buffer()
        #stations.to(opt.device)
        #print(stations.is_cuda)
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        player_x, player_y = hp.get_play_location()
        hornet_x, hornet_y = hp.get_hornet_location()
        soul = hp.get_souls()

        # 判定大黄蜂技能1
        hornet_skill1 = False
        if last_hornet_y > 32 and last_hornet_y < 32.5 and hornet_y > 32 and hornet_y < 32.5:
            hornet_skill1 = True
        #hornet_skill1 = False
        # 学习并输出
        move, action = agent.sample(stations, soul, hornet_x, hornet_y, player_x, hornet_skill1)

        take_direction(move)
        #print("Move"+Directions[move])
        take_action(action)
        #print("Action" + Actions[action])

        next_station = thread1.get_buffer()
        #next_station = next_station.to(opt.device)
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()


        # get reward
        move_reward = utilize.Helper.move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x,
                                             move, hornet_skill1)
        # print(move_reward)
        act_reward, done = utilize.Helper.action_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp,
                                                    next_player_x, next_hornet_x, next_hornet_x, action, hornet_skill1)
        # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)

        DeleyMoveReward.append(move_reward)
        DeleyActReward.append(act_reward)
        DeleyStation.append(stations)
        DeleyActions.append(action)
        DeleyDirection.append(move)
        DeleyStation.append(next_station)

        if len(DeleyStation) >= opt.DELEY_REWARD + 1:
            if DeleyMoveReward[0] != 0:
                move_rmp_correct.append((DeleyStation[0],DeleyDirection[0],DeleyMoveReward[0],DeleyStation[1],done))
            # if DeleyMoveReward[0] <= 0:
            #     move_rmp_wrong.append((DeleyStation[0],DeleyDirection[0],DeleyMoveReward[0],DeleyStation[1],done))

        if len(DeleyStation) >= opt.DELEY_REWARD + 1:
            if DeleyActReward[0] != 0:
                act_rmp_correct.append((DeleyStation[0],DeleyActions[0],DeleyActReward[0],DeleyStation[1],done))
            # if mean(DeleyActReward) <= 0:
            #     act_rmp_wrong.append((DeleyStation[0],DeleyActions[0],mean(DeleyActReward),DeleyStation[1],done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value

        total_reward += act_reward
        #paused = False
        #paused = utilize.Helper.pause_game(paused)
        # done 1 自己死了
        if done == 1:
            utilize.Actions.Nothing()
            break
        # done 2 BOSS死了
        elif done == 2:
            PASS_COUNT += 1
            utilize.Actions.Nothing()
            time.sleep(3)
            break

    thread1.stop()
    #batch_station, batch_actions, batch_reward, batch_next_station, batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)

    # 死亡后学习单次对战

    for i in range(8):
        if (len(move_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(opt.BATCH_SIZE)
            algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done,opt.BATCH_SIZE)

        if (len(act_rmp_correct) > opt.MEMORY_WARMUP_SIZE):
            print("action learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(opt.BATCH_SIZE)
            algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done,opt.BATCH_SIZE)


    return total_reward, step, PASS_COUNT, self_hp, boss_hp_value


>>>>>>> d00f7fee553731b8aa5db54bf247e6437c5d6052
