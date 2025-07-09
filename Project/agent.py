import torch, random
import numpy as np
import matplotlib

from matplotlib.pyplot import plot
from model import Linear_QNet, QTrainer
from snakegame import Direction, Point, SnakeGame    


from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, snakegame):
        head = snakegame.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snakegame.direction   == Direction.LEFT
        dir_r = snakegame.direction == Direction.RIGHT
        dir_u = snakegame.direction == Direction.UP
        dir_d = snakegame.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snakegame.is_collision(point_r)) or
            (dir_l and snakegame.is_collision(point_l)) or
            (dir_u and snakegame.is_collision(point_u)) or
            (dir_d and snakegame.is_collision(point_d)),

            # Danger right
            (dir_u and snakegame.is_collision(point_u)) or
            (dir_d and snakegame.is_collision(point_d)) or
            (dir_l and snakegame.is_collision(point_l)) or
            (dir_r and snakegame.is_collision(point_r)),

            # Danger left
            (dir_d and snakegame.is_collision(point_d)) or
            (dir_u and snakegame.is_collision(point_u)) or
            (dir_r and snakegame.is_collision(point_r)) or
            (dir_l and snakegame.is_collision(point_l)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            snakegame.food.x < snakegame.head.x, # food left
            snakegame.food.x > snakegame.head.x, # food right
            snakegame.food.y < snakegame.head.y, # food up
            snakegame.food.y > snakegame.head.y, # food down
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: tradeoff eploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snakegame = SnakeGame()

    while True:
        #get old state
        state_old = agent.get_state(snakegame)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = snakegame.play_step(final_move, record)
        state_new = agent.get_state(snakegame)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory,plot
            snakegame.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            #plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()