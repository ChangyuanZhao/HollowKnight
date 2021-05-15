# Define the actions we may need during training
# You can define your actions here

from utilize.SendKey import PressKey, ReleaseKey
from utilize.WindowsAPI import grab_screen
import time
import cv2
import threading


# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LEFT_ARROW = 0x25
RIGHT_ARROW = 0x27


# 回复 Cure
A = 0x41
# 跳跃 Jumo
Z = 0x5A
# 攻击 Attack
X = 0x58
# 冲刺 Rush
C = 0x43
# 超级冲刺 SuperRush
S = 0x53
# 技能 Skill
F = 0x46


# move actions
# 0
def Nothing():
    ReleaseKey(LEFT_ARROW)
    ReleaseKey(RIGHT_ARROW)
    pass


# Move
# 0
def Move_Left():
    PressKey(LEFT_ARROW)
    time.sleep(0.01)


# 1
def Move_Right():
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)


# 2
def Turn_Left():
    PressKey(LEFT_ARROW)
    time.sleep(0.01)
    ReleaseKey(LEFT_ARROW)


# 3
def Turn_Right():
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)
    ReleaseKey(RIGHT_ARROW)


# ----------------------------------------------------------------------

# other actions
# Attack
# 0
def Attack():
    PressKey(X)
    time.sleep(0.15)
    ReleaseKey(X)
    Nothing()
    time.sleep(0.01)


# 1
# def Attack_Down():
#     PressKey(DOWN_ARROW)
#     PressKey(X)
#     time.sleep(0.05)
#     ReleaseKey(X)
#     ReleaseKey(DOWN_ARROW)
#     time.sleep(0.01)
# 1
def Attack_Up():
    # print("Attack up--->")
    PressKey(UP_ARROW)
    PressKey(X)
    time.sleep(0.11)
    ReleaseKey(X)
    ReleaseKey(UP_ARROW)
    Nothing()
    time.sleep(0.01)


# JUMP
# 2
def Short_Jump():
    PressKey(Z)
    PressKey(DOWN_ARROW)
    PressKey(X)
    time.sleep(0.2)
    ReleaseKey(X)
    ReleaseKey(DOWN_ARROW)
    ReleaseKey(Z)
    Nothing()


# 3
def Mid_Jump():
    PressKey(Z)
    time.sleep(0.2)
    PressKey(X)
    time.sleep(0.2)
    ReleaseKey(X)
    ReleaseKey(Z)
    Nothing()


# Skill
# 4
# def Skill():
#     PressKey(Z)
#     PressKey(X)
#     time.sleep(0.1)
#     ReleaseKey(Z)
#     ReleaseKey(X)
#     time.sleep(0.01)
# 4
def Skill_Up():
    PressKey(UP_ARROW)
    PressKey(F)
    PressKey(X)
    time.sleep(0.15)
    ReleaseKey(UP_ARROW)
    ReleaseKey(F)
    ReleaseKey(X)
    Nothing()
    time.sleep(0.15)


# 5
def Skill_Down():
    PressKey(DOWN_ARROW)
    PressKey(F)
    PressKey(X)
    time.sleep(0.2)
    ReleaseKey(F)
    ReleaseKey(DOWN_ARROW)
    ReleaseKey(Z)
    Nothing()
    time.sleep(0.3)


# Rush
# 6
def Rush():
    PressKey(C)
    time.sleep(0.1)
    ReleaseKey(C)
    Nothing()
    PressKey(X)
    time.sleep(0.03)
    ReleaseKey(X)


# Cure
def Cure():
    PressKey(A)
    time.sleep(1.4)
    ReleaseKey(A)
    time.sleep(0.1)


# Restart function
# it restart a new game
# it is not in actions space
def Look_up():
    PressKey(UP_ARROW)
    time.sleep(0.1)
    ReleaseKey(UP_ARROW)

# 用来重置玩家位置，并开始战斗
def restart():

    while True:
        station = cv2.resize(cv2.cvtColor(grab_screen(), cv2.COLOR_RGBA2RGB), (1000, 500))
        if station[187][300][0] != 0:
            time.sleep(1)
        else:
            break
    time.sleep(1)
    Look_up()
    time.sleep(1.5)
    Look_up()
    time.sleep(1)
    while True:
        # 输出格式为 宽 + 高
        station = cv2.resize(cv2.cvtColor(grab_screen(), cv2.COLOR_RGBA2RGB), (1000, 500))
#        print(station.shape())

        # 对应像素点为白色即选择
        if station[238][586][0] > 200:
            # PressKey(DOWN_ARROW)
            # time.sleep(0.1)
            # ReleaseKey(DOWN_ARROW)
            PressKey(Z)
            time.sleep(0.1)
            ReleaseKey(Z)
            break
        else:
            Look_up()
            time.sleep(0.2)


# List for action functions
Actions = [Attack, Attack_Up,
           Short_Jump, Mid_Jump, Skill_Up,
           Skill_Down, Rush, Cure]
Directions = [Move_Left, Move_Right, Turn_Left, Turn_Right]


# Run the action
def take_action(action):
    Actions[action]()


def take_direction(direc):
    Directions[direc]()


class TackAction(threading.Thread):
    def __init__(self, threadID, name, direction, action):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.direction = direction
        self.action = action

    def run(self):
        take_direction(self.direction)
        take_action(self.action)