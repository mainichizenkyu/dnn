# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:07:41 2019

@author: amanuma_yuta
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from matplotlib import animation
from keras.models import Model
from keras import backend as K

#迷路の実装
#迷路でかいやつ
fig = plt.figure(figsize=(10, 14))
ax = plt.gca()

plt.plot([1, 2], [6, 6], color='red', linewidth=2)
plt.plot([2, 2], [6, 5], color='red', linewidth=2)
plt.plot([2, 4], [5, 5], color='red', linewidth=2)
plt.plot([3, 5], [6, 6], color='red', linewidth=2)
plt.plot([4, 4], [4, 5], color='red', linewidth=2)
plt.plot([3, 5], [4, 4], color='red', linewidth=2)
plt.plot([0, 1], [3, 3], color='red', linewidth=2)
plt.plot([1, 1], [2, 5], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 4], color='red', linewidth=2)
plt.plot([0, 3], [1, 1], color='red', linewidth=2)
plt.plot([3, 3], [1, 3], color='red', linewidth=2)
plt.plot([3, 4], [3, 3], color='red', linewidth=2)
plt.plot([4, 4], [1, 2], color='red', linewidth=2)
plt.plot([4, 5], [1, 1], color='red', linewidth=2)

plt.text(0.5, 6.5, 'S0', size=14, ha='center')
plt.text(1.5, 6.5, 'S1', size=14, ha='center')
plt.text(2.5, 6.5, 'S2', size=14, ha='center')
plt.text(3.5, 6.5, 'S3', size=14, ha='center')
plt.text(4.5, 6.5, 'S4', size=14, ha='center')
plt.text(0.5, 5.5, 'S5', size=14, ha='center')
plt.text(1.5, 5.5, 'S6', size=14, ha='center')
plt.text(2.5, 5.5, 'S7', size=14, ha='center')
plt.text(3.5, 5.5, 'S8', size=14, ha='center')
plt.text(4.5, 5.5, 'S9', size=14, ha='center')
plt.text(0.5, 4.5, 'S10', size=14, ha='center')
plt.text(1.5, 4.5, 'S11', size=14, ha='center')
plt.text(2.5, 4.5, 'S12', size=14, ha='center')
plt.text(3.5, 4.5, 'S13', size=14, ha='center')
plt.text(4.5, 4.5, 'S14', size=14, ha='center')
plt.text(0.5, 3.5, 'S15', size=14, ha='center')
plt.text(1.5, 3.5, 'S16', size=14, ha='center')
plt.text(2.5, 3.5, 'S17', size=14, ha='center')
plt.text(3.5, 3.5, 'S18', size=14, ha='center')
plt.text(4.5, 3.5, 'S19', size=14, ha='center')
plt.text(0.5, 2.5, 'S20', size=14, ha='center')
plt.text(1.5, 2.5, 'S21', size=14, ha='center')
plt.text(2.5, 2.5, 'S22', size=14, ha='center')
plt.text(3.5, 2.5, 'S23', size=14, ha='center')
plt.text(4.5, 2.5, 'S24', size=14, ha='center')
plt.text(0.5, 1.5, 'S25', size=14, ha='center')
plt.text(1.5, 1.5, 'S26', size=14, ha='center')
plt.text(2.5, 1.5, 'S27', size=14, ha='center')
plt.text(3.5, 1.5, 'S28', size=14, ha='center')
plt.text(4.5, 1.5, 'S29', size=14, ha='center')
plt.text(0.5, 0.5, 'S30', size=14, ha='center')
plt.text(1.5, 0.5, 'S31', size=14, ha='center')
plt.text(2.5, 0.5, 'S32', size=14, ha='center')
plt.text(3.5, 0.5, 'S33', size=14, ha='center')
plt.text(4.5, 0.5, 'S34', size=14, ha='center')

plt.text(0.5, 6.3, 'START', ha='center')
plt.text(0.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 5)
ax.set_ylim(0, 7)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5], [6.5], marker="o", color='g', markersize=60)
#行動可能か否かを示す, nanは不可　1 は可能, ↑, →, ↓, ←　で配置されている
theta_0=np.array([[np.nan, 1, 1,np.nan],#s0
                  [np.nan, 1, np.nan, 1],#s1
                  [np.nan, 1, 1, 1],#s2
                  [np.nan, 1, np.nan, 1],#s3
                  [np.nan, np.nan, np.nan, 1],#s4
                  [1, 1, 1, np.nan],#s5
                  [np.nan, np.nan, 1, 1],#s6
                  [1, 1, np.nan, np.nan],#s7
                  [np.nan, 1, np.nan, 1],#s8
                  [np.nan, np.nan, 1, 1],#s9
                  [1, np.nan, 1, np.nan],#s10
                  [1, 1, 1, np.nan],#s11
                  [np.nan, 1, 1, 1],#s12
                  [np.nan, np.nan, np.nan,1],#s13
                  [1, np.nan, np.nan, np.nan],#14
                  [1, np.nan, np.nan, np.nan],#15
                  [1, np.nan, 1, np.nan],#16
                  [1, 1, 1, np.nan],#17
                  [np.nan, 1, np.nan, 1],#18
                  [np.nan, np.nan, 1, 1],#19
                  [np.nan, np.nan, 1, np.nan],#20
                  [1, np.nan ,np.nan, np.nan],#21
                  [1, np.nan, 1, np.nan],#22
                  [np.nan, 1, 1, np.nan],#23
                  [1, np.nan, 1, 1],#24
                  [1 ,1 , np.nan, np.nan],#25
                  [np.nan, 1, np.nan, 1],#26
                  [1, np.nan, np.nan,1],#27
                  [1, np.nan, 1, np.nan],#28
                  [1, np.nan, np.nan, np.nan],#29
                  [np.nan, 1, np.nan, np.nan],#30
                  [np.nan, 1, np.nan, 1],#31
                  [np.nan, 1, np.nan, 1],#32
                  [1, 1, np.nan, 1],#33
                  [np.nan, np.nan, np.nan, 1],#34
                  ])

'''


fig = plt.figure(figsize=(8, 8))
ax = plt.gca()

plt.plot([0, 2], [3, 3], color='red', linewidth=2)
plt.plot([0, 2], [2, 2], color='red', linewidth=2)
plt.plot([3, 3], [2, 4], color='red', linewidth=2)
plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([2, 4], [1, 1], color='red', linewidth=2)

plt.text(0.5, 3.5, 'S0', size=14, ha='center')
plt.text(1.5, 3.5, 'S1', size=14, ha='center')
plt.text(2.5, 3.5, 'S2', size=14, ha='center')
plt.text(3.5, 3.5, 'S3', size=14, ha='center')
plt.text(0.5, 2.5, 'S4', size=14, ha='center')
plt.text(1.5, 2.5, 'S5', size=14, ha='center')
plt.text(2.5, 2.5, 'S6', size=14, ha='center')
plt.text(3.5, 2.5, 'S7', size=14, ha='center')
plt.text(0.5, 1.5, 'S8', size=14, ha='center')
plt.text(1.5, 1.5, 'S9', size=14, ha='center')
plt.text(2.5, 1.5, 'S10', size=14, ha='center')
plt.text(3.5, 1.5, 'S11', size=14, ha='center')
plt.text(0.5, 0.5, 'S12', size=14, ha='center')
plt.text(1.5, 0.5, 'S13', size=14, ha='center')
plt.text(2.5, 0.5, 'S14', size=14, ha='center')
plt.text(3.5, 0.5, 'S15', size=14, ha='center')

plt.text(0.5, 3.3, 'START', ha='center')
plt.text(3.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5], [6.5], marker="o", color='g', markersize=60)
#行動可能か否かを示す, nanは不可　1 は可能, ↑, →, ↓, ←　で配置されている
theta_0=np.array([[np.nan, 1, np.nan ,np.nan],#s0
                  [np.nan, 1, np.nan, 1],#s1
                  [np.nan, np.nan, 1, 1],#s2
                  [np.nan, np.nan, 1 , np.nan],#s3
                  [np.nan, 1, np.nan, np.nan],#s4
                  [np.nan, 1, np.nan, 1],#s5
                  [1, np.nan, 1, 1],#s6
                  [1, np.nan, 1, np.nan],#s7
                  [np.nan, 1, 1, np.nan],#s8
                  [np.nan, 1, 1, 1],#s9
                  [1, 1, np.nan, 1],#s10
                  [1, np.nan, np.nan, 1],#s11
                  [1, np.nan, np.nan, np.nan],#s12
                  [1, 1, np.nan, np.nan],#s13
                  [np.nan, 1, np.nan, 1],#14
                  [np.nan, np.nan, np.nan, 1],#15
                  ])






'''








def simple_convert_info_pi_from_theta(theta):#等確率に動く場合を考える
    
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0,m):
        pi[i, :] =theta[i, :]/ np.nansum(theta[i, :])
        
    pi = np.nan_to_num(pi)
    
    return pi

pi_0 = simple_convert_info_pi_from_theta(theta_0)




def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p = pi[s, :])
    
    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction == "down":
        s_next = s+3
    elif next_direction == "left":
        s_next = s-1
    return s_next

def goal_maze(pi):
    s = 0
    state_history = [0]
    
    while(1):
        next_s = get_action_and_next_s(pi, s)
        state_history.append(next_s)
        
        if next_s == 30:
            break
        else:
            s = next_s
            
    return state_history

#state_history = goal_maze(pi_0)
#print(state_history)
#print("迷路を解くのにかかったステップ数は"+str(len(state_history)-1)+"です")

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    state = state_history[i]
    x = (state % 5) + 0.5
    y = 6.5 - int(state / 5)
    line.set_data(x, y)
    return (line,)



#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history),interval=200,repeat=False)
#plt.show()
#anim.save("maze.gif", writer="imagemagick")






#ここからQ学習
def get_action(s, Q, epsilon,pi_0):#行動の取得
    direction = ["up", "right", "down", "left"]
    if np.random.rand()<epsilon:#epsilonの確率でランダムに動く
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:#行動価値関数の値が最大の行動を行う
        next_direction = direction[np.nanargmax(Q[s, :])]
    
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3
    return action


def get_s_next(s, a, Q, epsilon, pi):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]
    
    if next_direction == "up":
        s_next= s-5
    elif next_direction =="right":
        s_next= s+1
    elif next_direction =="down":
        s_next = s+5
    elif next_direction =="left":
        s_next =s-1
        
    return s_next

def Q_learning(s, a, r, s_next, Q, eta, gamma):#Q学習に基づく行動価値関数の更新
    if s_next == 30:#ゴールした場合
        Q[s, a] = Q[s, a] + eta*(r-Q[s, a])
    else:
        Q[s, a] = Q[s, a]+eta*(r+ gamma*np.nanmax(Q[s_next,:])-Q[s, a])#nanmax関数はnanを除いた配列の最大値を返す
    return Q

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):#迷路を解く
    s = 0#初期状態
    a = a_next = get_action(s, Q, epsilon, pi) #最初の行動の取得
    s_a_history = [[0, np.nan]]#エージェントの行動を記録するリスト
    while(1):
        a = a_next
        s_a_history[-1][1] = a
        #現在の状態に行動を代入
        s_next = get_s_next(s, a, Q, epsilon, pi)#現在の状態で行動aを取った場合の次の状態の取得
        s_a_history.append([s_next, np.nan])
        
        if s_next == 30:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next= get_action(s_next, Q, epsilon, pi)
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        
        if s_next == 30:
            break
        else:
            s = s_next
            
    return[s_a_history, Q]
        
        
        
    
#Q関数の初期値の定義
[a, b]= theta_0.shape#状態数aと行動パターンbの数を取得する
Q = np.random.rand(a, b)*theta_0   #theta_0で不能な行動はnan, それ以外はランダムに決定

#Q学習で迷路の最適化を実施
eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1)#状態ごとに最大価値を抽出
is_continue = True
episode = 1
V = []
V.append(np.nanmax(Q, axis=1))
step=[]
while is_continue:
    print("エピソード:"+str(episode))
    epsilon = epsilon / 1.2
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon ,eta, gamma, pi_0)
    new_V = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_V-v)))
    v = new_V
    V.append(v)
    
    print("迷路を解くのにかかったステップ数は" + str(len(s_a_history)-1)+"です")
    episode = episode + 1
    step.append(len(s_a_history))
    if episode > 100:
        break
s_a_history = np.array(s_a_history)
state_history = s_a_history[:, 0]
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history),interval=200,
                               repeat=False)
plt.show()
anim.save("maze.gif", writer="imagemagick")

#収束過程の図示
plt.figure(figsize=(12, 6))
plt.rcParams["font.size"] = 20
plt.plot(step)
plt.xlabel('episode')
plt.ylabel('number of step')




'''

#ここから先はDQNの実装

from collections import namedtuple

Transition = namedtuple('Transition' , ('state', 'action', 'next_state', 'reward', 'movables'))

GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500
CAPACITY = 1000

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity =CAPACITY
        self.memory = []
        self.index = 0
        
    def push(self, state, action, next_state, reward, movables):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.index] = Transition(state, action, next_state, reward, movables)
            self.index = (self.index + 1) % self.capacity
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 64
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  #行動の数を取得
        self.epsilon = 0.99
        self.e_decay = 0.9
        self.e_min = 0.001
        self.memory = ReplayMemory(CAPACITY)
        #状態及び行動を入れると状態行動価値が出力される1入力多出力のニューラルネットワーク
        self.model = Sequential()
        self.model.add(Dense(32, input_shape = (2, 2), activation='tanh'))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation = 'tanh'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999))
        
    def replay(self):
        #メモリサイズがミニバッチサイズ以下の時は何もしない
        if len(self.memory) < BATCH_SIZE:
            return
        #バッチサイズだけデータの取り出し
        transitions = self.memory.sample(BATCH_SIZE)
        #データの形状を変形する
        batch = Transition(*zip(*transitions))
        #変数の取り出し
        state_batch = np.array(batch.state)
        next_s_batch = np.array(batch.next_state)
        reward_batch = np.array(batch.reward)
        action_batch = np.array(batch.action)
        movables_batch = np.array(batch.movables)
        X =[]
        Y =[]
        for i in range(BATCH_SIZE):
            state = state_batch[i]
            action = action_batch[i]
            next_s = next_s_batch[i]
            reward = reward_batch[i]
            next_movables = movables_batch[i]
            next_rewards = []
            for j in range(4):
                if next_movables[j] == None:
                    #行動できないパターンについては何もしない
                    pass
                else:
                    #行動できる場合については対応する行動を生成
                    if j == 0:
                        action = [0, 1]
                    elif j == 1:
                        action = [1, 0]
                    elif j == 2:
                        action = [0, -1]
                    elif j == 3:
                        action = [-1, 0]
                    next_s=np.array(next_s)
                    next_a=np.array(action)
                    next_s_a=np.array([[next_s, next_a]])#状態と行動をセットにする
                    next_rewards.append(self.model.predict(next_s_a))#現在のニューラルネットワークで報酬の出力
            next_rewards_max = np.max(next_rewards)#最大値の取得
            target_f = reward + GAMMA*next_rewards_max#教師データを生成
            X.append([state, action])
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        self.model.fit(np_X, np_Y, epochs =16)#TD誤差が最小になるようにしているのと同値
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay
        print(self.epsilon)
        
    def decide_action(self, state, s, movables):
        if self.epsilon <= np.random.uniform(0, 1):#ε-greedy法
            action_value = []
            act = []
            for j in range(4):    
                if movables[j] == None:
                    pass
                else:
                    if j == 0:
                        action = [0, 1]
                    elif j == 1:
                        action = [1, 0]
                    elif j == 2:
                        action = [0, -1]
                    elif j == 3:
                        action = [-1, 0]
                    state = np.array(state)
                    action = np.array(action)
                    state_action = np.array([[state, action]])
                    action_value.append(self.model.predict(state_action))#行動価値の取得
                    act.append(j)#対応する行動の記録
            action_value = np.array(action_value)
            action_pre = np.argmax(action_value)#価値が最大のインデックスの取得
            action = act[action_pre]#価値が最大の場合の行動の取得
        else:
            direction = np.array([i for i in range(4)])
            action = np.random.choice(direction, p=pi_0[s, :])
        return action

    def decide_action_test(self, state, s, movables):
        action_value = []
        act = []
        for j in range(4):    
            if movables[j] == None:
                pass
            else:
                if j == 0:
                    action = [0, 1]
                elif j == 1:
                    action = [1, 0]
                elif j == 2:
                    action = [0, -1]
                elif j == 3:
                    action = [-1, 0]
                state = np.array(state)
                action = np.array(action)
                state_action = np.array([[state, action]])
                action_value.append(self.model.predict(state_action))
                act.append(j)
        action_value = np.array(action_value)
        action_pre = np.argmax(action_value)
        action = act[action_pre]
        return action, action_value, act    

#迷路をDQNを用いて解く
        

def get_s_next(state ,s, a):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]
    s_next = state
    if next_direction == "up":
        s_next[1] += 1
        s -= 5
    elif next_direction =="right":
        s_next[0] += 1
        s += 1
    elif next_direction =="down":
        s_next[1] -= 1
        s += 5
    elif next_direction =="left":
        s_next[0] -= 1
        s -= 1
    return s_next, s    
    

#行動できない部分をNoneに置き換えた配列の生成
dqn_pi = np.where(pi_0 == 0, None, pi_0)
state_size = 1
action_size = 1
dqn_solver = Brain(state_size, action_size)

episodes = 10
times = 0
for e in range(episodes):
    state = [0.5, 6.5]
    s = 0
    while(1):
        times+=1
        movables = dqn_pi[s]
        action = dqn_solver.decide_action(np.array(state),s , movables)
        next_state,next_s = get_s_next(np.array(state), s, action)
        next_movables = dqn_pi[next_s]
        if next_s == 6:
            reward = 1
        else :
            reward = 0
        dqn_solver.memory.push(np.array(state), action, np.array(next_state), reward, next_movables)
        if next_s == 6:
            break
        s=next_s
        state = np.array(next_state)        
    dqn_solver.replay()
    

    
def goal_maze(pi):
    s_test = 0
    state_test = [0.5, 6.5]
    state_history = [0]
    num=0
    while(1):
        num+=1
        movables_test = dqn_pi[s_test]
        action_test, action_value , act= dqn_solver.decide_action_test(np.array(state_test), np.array(s_test), movables_test)
        next_state_test, next_s_test = get_s_next(np.array(state_test), s_test,  action_test)
        state_history.append(next_s_test)
        if next_s_test == 6:
            break
        elif num == 10000:
            break
        else:
            s_test = next_s_test
            state_test = next_state_test
            
    return state_history

state_history = goal_maze(pi_0)
print("迷路を解くのにかかったステップ数は"+str(len(state_history)-1)+"です")





def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    state = state_history[i]
    x = (state % 3) + 0.5
    y = 2.5 - int(state / 3)
    line.set_data(x, y)
    return (line,)



anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history),interval=200,
                               repeat=False)
plt.show()
anim.save("maze.gif", writer="imagemagick")
'''
