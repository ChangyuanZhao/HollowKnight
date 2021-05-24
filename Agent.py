import numpy as np
import torch
import time

class Agent:
    def __init__(self, act_dim, algorithm, opt, e_greed=0.1, e_greed_rate=0.99):
        self.act_dim = act_dim
        self.move_dim = opt.movedim
        self.algorithm = algorithm
        self.e_greed = e_greed
        self.e_greed_rate = e_greed_rate
        self.device = opt.device

    def sample(self, station, soul, hornet_x, hornet_y, player_x, hornet_skill1):

        #lasttime = time.time()
        pred_move = self.algorithm.move_model(station.to(self.device))
        #print(pred_move.get_device())
        #print("move time" + str(time.time() - lasttime))
        #lasttime = time.time()
        pred_act = self.algorithm.act_model(station.to(self.device))
        #print("act time" + str(time.time() - lasttime))
        # print(pred_move)
        # print(self.e_greed)
        pred_move = pred_move.cpu().detach().numpy()
        pred_act = pred_act.cpu().detach().numpy()

        # 预测移动
        sample = np.random.rand()
        if sample < self.e_greed:

            move = self.better_move(hornet_x, player_x, hornet_skill1)
        else:

            # 返回最大值索引
            move = np.argmax(pred_move)

        self.e_greed = self.e_greed * self.e_greed_rate

        # 预测动作
        sample = np.random.rand()
        if sample < self.e_greed:
            act = self.better_action(soul, hornet_x, hornet_y, player_x, hornet_skill1)
        else:
            act = np.argmax(pred_act)

        return move, act

    # 一定概率以 自定的策略移动
    def better_move(self, hornet_x, player_x, hornet_skill1):

        return np.random.randint(self.move_dim)
        '''
        dis = abs(player_x - hornet_x)
        dire = player_x - hornet_x
        if hornet_skill1:
            # run away while distance < 6
            if dis < 6:
                if dire > 0:
                    return 1
                else:
                    return 0
            # do not do long move while distance > 6
            else:
                if dire > 0:
                    return 2
                else:
                    return 3

        if dis < 2.5:
            if dire > 0:
                return 1
            else:
                return 0
        elif dis < 5:
            if dire > 0:
                return 2
            else:
                return 3
        else:
            if dire > 0:
                return 0
            else:
                return 1
        '''

    # 一定概率以 自定的策略行动
    def better_action(self, soul, hornet_x, hornet_y, player_x, hornet_skill1):
        '''
        dis = abs(player_x - hornet_x)
        if hornet_skill1:
            if dis < 3:
                return 6
            else:
                return 1
        '''
        if soul >= 33:
            return np.random.randint(self.act_dim)
        return np.random.randint(self.act_dim - 1)


