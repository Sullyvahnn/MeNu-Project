from collections import deque

import numpy as np
import torch

from game import *
from helper import plot
from model import Linear_Qnet, QTrainer

MAX_MEMORY = 1000000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 256


class Agent:
    def __init__(self):
        self.generation = 0
        # random move chance
        self.epsilon = 0
        # learning speed
        self.gamma = 0.9
        # I use deque bcs it automatically removes left elements if memory exceeded
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, HIDDEN_LAYER_SIZE, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def getState(self, game):
        # [danger left, danger straight, danger right
        # [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN,
        # Food.LEFT, Food.RIGHT, Food.UP, Food.DOWN
        head = game.snake[0]
        # points around the head
        head_left = Point(head.x - BLOCK_SIZE, head.y)
        head_right = Point(head.x + BLOCK_SIZE, head.y)
        head_down = Point(head.x, head.y + BLOCK_SIZE)
        head_up = Point(head.x, head.y - BLOCK_SIZE)
        # direction mark
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_down = game.direction == Direction.DOWN
        direction_up = game.direction == Direction.UP
        # food mark
        food_left = game.food.x < head.x
        food_right = game.food.x > head.x
        food_down = game.food.y > head.y
        food_up = game.food.y < head.y
        # collisions check
        straight_collision = ((game.direction == Direction.LEFT and game.is_collision(head_left)) or
                              (game.direction == Direction.RIGHT and game.is_collision(head_right)) or
                              (game.direction == Direction.DOWN and game.is_collision(head_down)) or
                              (game.direction == Direction.UP and game.is_collision(head_up)))
        left_collision = ((game.direction == Direction.LEFT and game.is_collision(head_down)) or
                          (game.direction == Direction.RIGHT and game.is_collision(head_up)) or
                          (game.direction == Direction.DOWN and game.is_collision(head_right)) or
                          (game.direction == Direction.UP and game.is_collision(head_left)))
        right_collision = ((game.direction == Direction.LEFT and game.is_collision(head_up)) or
                           (game.direction == Direction.RIGHT and game.is_collision(head_down)) or
                           (game.direction == Direction.DOWN and game.is_collision(head_left)) or
                           (game.direction == Direction.UP and game.is_collision(head_right)))
        state = np.array([straight_collision, right_collision, left_collision,
                          direction_left, direction_right, direction_up, direction_down,
                          food_left, food_right, food_up, food_down])
        return state

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def trainLongMemory(self):
        # take part of memory if > BATH_SIZE
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples len BATCH_SIZE
        else:
            mini_sample = self.memory
        # prepare arguments of trainStep function
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, game_overs)  # one argument aggregated

    def trainShortMemory(self, state, action, reward, next_state, game_over):
        self.trainer.trainStep(state, action, reward, next_state, game_over)

    def getAction(self, state):
        self.epsilon = 200 - self.generation
        final_move = [0, 0, 0]
        # first moves should be random
        # later less exploration more exploitation
        if random.randint(0, 300) < self.epsilon:
            move_index = random.randint(0, 2)
            final_move[move_index] = 1
        else:
            #prepare data and make prediction based on the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_index = torch.argmax(prediction).item()
            final_move[move_index] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get agent state
        state_old = agent.getState(game)
        # calculate new agent state
        final_move = agent.getAction(state_old)
        # perform the move
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.getState(game)
        #train short memory
        agent.trainShortMemory(state_old, final_move, reward, state_new, game_over)
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory
            game.reset()
            agent.generation += 1
            agent.trainLongMemory()
            if score > best_score:
                best_score = score
                agent.model.save()
            print("Generation: ", agent.generation, "Score: ", score, "Best score: ", best_score)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.generation
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
