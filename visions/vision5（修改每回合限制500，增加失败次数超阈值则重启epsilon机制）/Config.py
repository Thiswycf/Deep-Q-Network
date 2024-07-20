# DEBUG = True
DEBUG = False
# Hyperparameters
MEMORY_SIZE = 10**5
BATCH_SIZE = 64
EPSILON_START = 1.00
EPSILON_END = 0.01
if DEBUG:
    EPSILON_START = 0
    EPSILON_END = 0
EPSILON_DECAY = 500

# 越靠近1越远视，反之则越短视
GAMMA = 1 - 1e-4
LR = 0.01

# TAU = 20
TAU = 0.05
NN_WIDTH = 64
NN_DEPTH = 3

NUM_EPISODES = 600
# 期望目标是500
EXPECTION = 500
# 训练时的终止阈值设为1倍
STOP_THRESHOLD = EXPECTION

# 连续完成50次视为达到训练目标
CONTINUOUS_DONE_COUNTER_THRESHOLD = 50

# 连续失败100次视为到达一个新领域，需要重新递减epsilon
CONTINUOUS_LOSE_COUNTER_THRESHOLD = 100

ENV_NAME='CartPole-v1'
# ENV_NAME='MountainCar-v0'
