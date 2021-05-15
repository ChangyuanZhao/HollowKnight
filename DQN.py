<<<<<<< HEAD
from torch import nn
import torch

class DQN:
    def __init__(self, actnet, movenet, opt, gamma=0.9, learning_rate=0.0001):
        self.act_dim = actnet.actdim
        self.act_model = actnet
        self.move_model = movenet
        self.move_dim = movenet.movedim
        self.device = opt.device
        self.gamma = torch.tensor(gamma)
        self.lr = learning_rate


        #  模型的优化器 和 评级韩式
        self.act_model.optimizer = torch.optim.Adam(self.act_model.parameters(), lr=self.lr)
        self.act_model.loss_func = nn.MSELoss()

        self.move_model.optimizer = torch.optim.Adam(self.move_model.parameters(), lr=self.lr)
        self.move_model.loss_func = nn.MSELoss()

        self.act_global_step = 0
        self.move_global_step = 0
        self.update_target_steps = 100  # 每隔200个training steps再把model的参数复制到target_model中

    # train functions for act model
    def act_predict(self, obs):

        # 使用动作模型获取  [Q(s,a1),Q(s,a2),...]
        return self.act_model(obs)

    def act_train_step(self, action, features, label, next_obs):

        # 训练步骤
        # 动作预测值
        prediction = self.act_model(features.to(self.device)).cpu().gather(1, torch.tensor(action).reshape((1,1))).squeeze(-1)
        next_prediction = self.act_model(next_obs.to(self.device)).cpu().max(1)[0]
        reward = prediction + self.gamma * next_prediction
        label = torch.tensor(label).reshape((1, 1)).squeeze(-1)
        loss = self.act_model.loss_func(label, reward)
        return loss

    def act_train_model(self, action, features, labels, next_obs, batch_size):

        # 训练模型
        self.act_model.zero_grad()
        loss = torch.tensor(0.)
        for epoch in range(batch_size):
            loss = self.act_train_step(action[epoch], features[epoch], labels[epoch], next_obs[epoch])
        #print(loss)
        loss = loss/batch_size
        loss = loss.to(self.device)
        print("act loss : ", loss)
        loss.backward()
        self.act_model.optimizer.step()
        torch.cuda.empty_cache()

    def act_learn(self, obs, action, reward, next_obs, terminal, batch_size):

        # 使用DQN算法 更新数值网络

        # print('learning')
        # 每隔200个training steps同步一次model和target_model的参数
        # if self.act_global_step % self.update_target_steps == 0:
        #     self.act_replace_target()

        # 从target_model中获取 max Q' 的值，用于计算target_Q

        self.act_train_model(action, obs, reward, next_obs, batch_size)
        #print(reward)
        self.act_global_step += 1
        # print('finish')

    def act_replace_target(self):
        '''
        # 预测模型权重更新到 target模型中

       # for i, l in enumerate(
            #    self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
           # l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
            #    index=i).get_weights())
       # for i, l in enumerate(
        #        self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
      #      l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
       #         index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        #self.act_target_model.get_layer(index=1).get_layer(index=2).set_weights(
           # self.act_model.get_layer(index=1).get_layer(index=2).get_weights())

        # self.act_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.act_model.get_layer(index=1).get_layer(index=6).get_weights())
        '''
    # train functions for move_model

    def move_predict(self, obs):

        # 同上预测 移动 [Q(s,a1),Q(s,a2),...]
        return self.move_model.predict(obs)

    def move_train_step(self, action, features, label, next_obs):

        # 训练步骤
        # 动作预测值
        prediction = self.move_model(features.to(self.device)).cpu().gather(1, torch.tensor(action).reshape((1, 1))).squeeze(-1)
        next_prediction = self.move_model(next_obs.to(self.device)).cpu().max(1)[0]
        reward = prediction + self.gamma * next_prediction
        label = torch.tensor(label).reshape((1, 1)).squeeze(-1)
        loss = self.move_model.loss_func(label, reward)
        return loss

    def move_train_model(self, action, features, labels, next_obs, batch_size):
        # 训练模型
        self.move_model.zero_grad()
        loss = torch.tensor(0.)
        for epoch in range(batch_size):
            loss += self.move_train_step(action[epoch], features[epoch], labels[epoch], next_obs[epoch])
        loss = loss/batch_size
        loss = loss.to(self.device)
        print("move loss : ", loss)
        loss.backward()
        self.move_model.optimizer.step()
        torch.cuda.empty_cache()

    def move_learn(self, obs, action, reward, next_obs, terminal, batch_size):
        """ 使用DQN算法更新self.move_model的value网络
        """
        self.move_train_model(action, obs, reward, next_obs, batch_size)
        #print(reward)
        #print('1231231')
        #print(action)
        self.move_global_step += 1

    def move_replace_target(self):
        '''预测模型权重更新到target模型权重'''
        '''
        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
                index=i).get_weights())
        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
                index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_target_model.get_layer(index=1).get_layer(index=2).set_weights(
            self.move_model.get_layer(index=1).get_layer(index=2).get_weights())
        '''
        # self.move_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.move_model.get_layer(index=1).get_layer(index=6).get_weights())

    def replace_target(self):
        # print("replace target")

        # copy conv3d_1
        '''
        self.model.shared_target_model.get_layer(index=0).set_weights(
            self.model.shared_model.get_layer(index=0).get_weights())
        # copy batchnormalization_1
        self.model.shared_target_model.get_layer(index=1).set_weights(
            self.model.shared_model.get_layer(index=1).get_weights())

        # copy shard_resnet block
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=1).get_layer(index=i).get_weights())

        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_replace_target()
        self.act_replace_target()
=======
from torch import nn
import torch

class DQN:
    def __init__(self, actnet, movenet, opt, gamma=0.9, learning_rate=0.0001):
        self.act_dim = actnet.actdim
        self.act_model = actnet
        self.move_model = movenet
        self.move_dim = movenet.movedim
        self.device = opt.device
        self.gamma = torch.tensor(gamma)
        self.lr = learning_rate


        #  模型的优化器 和 评级韩式
        self.act_model.optimizer = torch.optim.Adam(self.act_model.parameters(), lr=self.lr)
        self.act_model.loss_func = nn.MSELoss()

        self.move_model.optimizer = torch.optim.Adam(self.move_model.parameters(), lr=self.lr)
        self.move_model.loss_func = nn.MSELoss()

        self.act_global_step = 0
        self.move_global_step = 0
        self.update_target_steps = 100  # 每隔200个training steps再把model的参数复制到target_model中

    # train functions for act model
    def act_predict(self, obs):

        # 使用动作模型获取  [Q(s,a1),Q(s,a2),...]
        return self.act_model(obs)

    def act_train_step(self, action, features, label, next_obs):

        # 训练步骤
        # 动作预测值
        prediction = self.act_model(features.to(self.device)).cpu().gather(1, torch.tensor(action).reshape((1,1))).squeeze(-1)
        next_prediction = self.act_model(next_obs.to(self.device)).cpu().max(1)[0]
        reward = prediction + self.gamma * next_prediction
        label = torch.tensor(label).reshape((1, 1)).squeeze(-1)
        loss = self.act_model.loss_func(label, reward)
        return loss

    def act_train_model(self, action, features, labels, next_obs, batch_size):

        # 训练模型
        self.act_model.zero_grad()
        loss = torch.tensor(0.)
        for epoch in range(batch_size):
            loss = self.act_train_step(action[epoch], features[epoch], labels[epoch], next_obs[epoch])
        #print(loss)
        loss = loss/batch_size
        loss = loss.to(self.device)
        print("act loss : ", loss)
        loss.backward()
        self.act_model.optimizer.step()
        torch.cuda.empty_cache()

    def act_learn(self, obs, action, reward, next_obs, terminal, batch_size):

        # 使用DQN算法 更新数值网络

        # print('learning')
        # 每隔200个training steps同步一次model和target_model的参数
        # if self.act_global_step % self.update_target_steps == 0:
        #     self.act_replace_target()

        # 从target_model中获取 max Q' 的值，用于计算target_Q

        self.act_train_model(action, obs, reward, next_obs, batch_size)
        #print(reward)
        self.act_global_step += 1
        # print('finish')

    def act_replace_target(self):
        '''
        # 预测模型权重更新到 target模型中

       # for i, l in enumerate(
            #    self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
           # l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
            #    index=i).get_weights())
       # for i, l in enumerate(
        #        self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
      #      l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
       #         index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        #self.act_target_model.get_layer(index=1).get_layer(index=2).set_weights(
           # self.act_model.get_layer(index=1).get_layer(index=2).get_weights())

        # self.act_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.act_model.get_layer(index=1).get_layer(index=6).get_weights())
        '''
    # train functions for move_model

    def move_predict(self, obs):

        # 同上预测 移动 [Q(s,a1),Q(s,a2),...]
        return self.move_model.predict(obs)

    def move_train_step(self, action, features, label, next_obs):

        # 训练步骤
        # 动作预测值
        prediction = self.move_model(features.to(self.device)).cpu().gather(1, torch.tensor(action).reshape((1, 1))).squeeze(-1)
        next_prediction = self.move_model(next_obs.to(self.device)).cpu().max(1)[0]
        reward = prediction + self.gamma * next_prediction
        label = torch.tensor(label).reshape((1, 1)).squeeze(-1)
        loss = self.move_model.loss_func(label, reward)
        return loss

    def move_train_model(self, action, features, labels, next_obs, batch_size):
        # 训练模型
        self.move_model.zero_grad()
        loss = torch.tensor(0.)
        for epoch in range(batch_size):
            loss += self.move_train_step(action[epoch], features[epoch], labels[epoch], next_obs[epoch])
        loss = loss/batch_size
        loss = loss.to(self.device)
        print("move loss : ", loss)
        loss.backward()
        self.move_model.optimizer.step()
        torch.cuda.empty_cache()

    def move_learn(self, obs, action, reward, next_obs, terminal, batch_size):
        """ 使用DQN算法更新self.move_model的value网络
        """
        self.move_train_model(action, obs, reward, next_obs, batch_size)
        #print(reward)
        #print('1231231')
        #print(action)
        self.move_global_step += 1

    def move_replace_target(self):
        '''预测模型权重更新到target模型权重'''
        '''
        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
                index=i).get_weights())
        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
                index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_target_model.get_layer(index=1).get_layer(index=2).set_weights(
            self.move_model.get_layer(index=1).get_layer(index=2).get_weights())
        '''
        # self.move_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.move_model.get_layer(index=1).get_layer(index=6).get_weights())

    def replace_target(self):
        # print("replace target")

        # copy conv3d_1
        '''
        self.model.shared_target_model.get_layer(index=0).set_weights(
            self.model.shared_model.get_layer(index=0).get_weights())
        # copy batchnormalization_1
        self.model.shared_target_model.get_layer(index=1).set_weights(
            self.model.shared_model.get_layer(index=1).get_weights())

        # copy shard_resnet block
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=1).get_layer(index=i).get_weights())

        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_replace_target()
        self.act_replace_target()
>>>>>>> d00f7fee553731b8aa5db54bf247e6437c5d6052
        '''