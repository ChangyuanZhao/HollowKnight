from utilize.WindowsAPI import key_check
import time
import torch
'''

Actions = ['Attack',
           'Short_Jump', 'Mid_Jump', 'Rush','Skill', 'Cure']
Directions = ['Move_Left', 'Move_Right', 'Turn_Left', 'Turn_Right','Turn_Up','Turn_Down']

'''
# check whether a game is end
def is_end(next_self_blood, min_hp, next_boss_blood, boss_blood):
    if next_self_blood == 9 and min_hp <= 3:
        return True
    elif next_boss_blood - boss_blood > 200:
        return True
    return False


# get mean score of a reward seq
def mean(d):
    t = 0
    for i in d:
        t += i
    return t / len(d)


# 自身HP reward   每次掉血扣50  不掉血扣 1（防止不停撞墙）
def count_self_reward(next_self_blood, self_hp):
    if next_self_blood - self_hp < 0:
        return 50 * (next_self_blood - self_hp)
    return -1

# 平A 掉21 法术20~30
# BOSS HP reward
def count_boss_reward(next_boss_blood, boss_blood):
    if next_boss_blood - boss_blood < 0:
        return int((boss_blood - next_boss_blood) * 4)
    return -1

# 方向reward
def direction_reward(move, player_x, hornet_x, player_y, hornet_y):
    if move == 0 or move == 1:
        return 0
    else:
        dirx = player_x - hornet_x
        diry = player_y - hornet_y
        if move == 2:
            if dirx > 0:
                return 20
            else:
                return -20
        elif move == 3:
            if dirx < 0:
                return 20
            else:
                return -20
        elif move == 4:
            if diry < 0:
                if abs(dirx) > 3:
                    return -10
                else:
                    return 20
            else:
                return -20
        elif move == 5:
            if diry > 0:
                if abs(dirx) > 3:
                    return -10
                else:
                    return 20
            else:
                return -20

# 距离reward
def distance_reward(move, next_player_x, next_hornet_x):
    if abs(next_player_x - next_hornet_x) < 1.5:
        return -6
    elif abs(next_player_x - next_hornet_x) < 4.8:
        return 20
    else:
        if move < 2:
            return 10
        else:
            return -5

# 移动reward判定
def move_judge(self_blood, next_self_blood, next_boss_blood, boss_blood, player_x, player_y, next_player_x, hornet_x, hornet_y, next_hornet_x, move, hornet_skill1):
    # reward = count_self_reward(next_self_blood, self_blood)
    # if reward < 0:
    #     return reward
    '''
    if hornet_skill1:
        # run away while distance < 5
        if abs(player_x - hornet_x) < 6:
            # change direction while hornet use skill
            if move == 0 or move == 2:
                dire = 1
            else:
                dire = -1
            if player_x - hornet_x > 0:
                s = -1
            else:
                s = 1
            # if direction is correct and use long move
            if dire * s == 1 and move < 2:
                return 10
        # do not do long move while distance > 5
        else:
            if move >= 2:
                return 10
        return -10
    '''
    '''
    dis = abs(player_x - hornet_x)
    dire = player_x - hornet_x
    if move == 0:
        if (dis > 5 and dire > 0) or (dis < 2.5 and dire < 0):
            return 10
    elif move == 1:
        if (dis > 5 and dire < 0) or (dis < 2.5 and dire > 0):
            return 10
    elif move == 2:
        if dis > 2.5 and dis < 5 and dire > 0:
            return 10
    elif move == 3:
        if dis > 2.5 and dis < 5 and dire < 0:
            return 10
    '''
# 没有跳跃reward
    reward = 0
    self_blood_reward = count_self_reward(next_self_blood, self_blood) + 10
    boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)/3
    dir_reward = direction_reward(move, player_x, hornet_x, player_y, hornet_y)
    reward = self_blood_reward + boss_blood_reward + dir_reward

    #reward += direction_reward(move, player_x, hornet_x)
    #reward += distance_reward(move, next_player_x, next_hornet_x)
    # reward = direction_reward(move, player_x, hornet_x) + distance_reward(move, player_x, hornet_x)
    return reward


def act_skill_reward(action, next_hornet_x, next_hornet_y, next_player_x):
    skill_reward = 0
    '''
    if hornet_skill1:
        if action == 2 or action == 3:
            skill_reward -= 5
    elif next_hornet_y > 34 and abs(next_hornet_x - next_player_x) < 5:
        if action == 4:
            skill_reward += 2
    '''
    # 技能reward

    return skill_reward


def act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y):
    distance_reward = 0
    if abs(next_player_x - next_hornet_x) < 12:
        if abs(next_player_x - next_hornet_x) > 6:
            if action >= 2 and action <= 3:
                # distance_reward += 0.5
                pass
            elif next_hornet_y < 29 and action == 6:
                distance_reward -= 3
        else:
            if action >= 2 and action <= 3:
                distance_reward -= 0.5
    else:
        if action == 0 and action == 1:
            distance_reward -= 3
        elif action == 6:
            distance_reward += 1
    return distance_reward


# JUDGEMENT FUNCTION, write yourself
def action_judge(soul, boss_blood, next_boss_blood, self_blood, next_self_blood, next_player_x, next_hornet_x, next_hornet_y,
                 action, hornet_skill1):
    rate = [1, 1, 1, 1, 2]
    self_blood_reward = count_self_reward(next_self_blood, self_blood) / 3
    boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
    reward = self_blood_reward + boss_blood_reward * rate[action]
    #reward *= rate[action]
    if soul < 33 and action == 4:
        reward -= 100

    # Player dead
    if next_self_blood <= 0 and self_blood != 9:
        '''
        #skill_reward = act_skill_reward(hornet_skill1, action, next_hornet_x, next_hornet_y, next_player_x)
        #distance_reward = act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)/4
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        #+ distance_reward + skill_reward
        #if action == 4:
         #   reward *= 1.5
        #elif action == 5:
        #    reward *= 0.5
        '''
        done = 1
        #return reward, done
    # boss dead

    elif next_boss_blood <= 0 or next_boss_blood > 900:
        '''
        #skill_reward = act_skill_reward(hornet_skill1, action, next_hornet_x, next_hornet_y, next_player_x)
        #distance_reward = act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)/4
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)


        reward = self_blood_reward + boss_blood_reward
        #+ distance_reward + skill_reward
        #if action == 4:
         #   reward *= 1.5
        #elif action == 5:
        #    reward *= 0.5
        '''
        done = 2
        #return reward, done
    # playing
    else:
        '''
        skill_reward = act_skill_reward(hornet_skill1, action, next_hornet_x, next_hornet_y, next_player_x)
        distance_reward = act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)/4
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        #reward = self_blood_reward + boss_blood_reward + distance_reward + skill_reward
        #if action == 4:
        #    reward *= 1.5
        #elif action == 5:
        #    reward *= 0.5
        '''
        done = 0
    return reward, done


# Paused training
def pause_game(paused, actmodel, movemodel, episode = 0):
    op, d = key_check()
    if 'T' in op:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            op, d = key_check()
            # pauses game and can get annoying.
            if 'T' in op:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
            if 'Save' in op:
                if paused:
                    print('save model')
                    torch.save(actmodel.state_dict(), 'model/act_%s'%episode)
                    torch.save(movemodel.state_dict(), 'model/move_%s'%episode)
                    time.sleep(1)


    return paused