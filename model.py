import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # creating DNN
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ReLU activation function
        x = F.relu(self.linear1(x))
        # pass it through sec layer
        x = self.linear2(x)
        return x

    # saves model to model.pth file
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# trains model
class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.lr = learning_rate
        # future rewards price
        self.gamma = gamma
        self.model = model
        # choosing optimization algorithm
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # choosing criterion as mean square errors loss sum(Q-Q_new)^2
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, next_state, game_over):
        # preparing states data as tensors
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = np.array(action)
        action = torch.tensor(action, dtype=torch.long)
        reward = np.array(reward)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # if working on short memory need to change dim of the data to match our needs
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # creating a tuple
            game_over = (game_over,)

        # 1: predicted future Q values with current state
        pred = self.model(state)

        target = pred.clone()
        # going through every element in tensor and calculate max future reward
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            # if game ended we calculate only present rewards
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # creating matrix of rewards for possible states
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # handling optimization algorithms

        # clear gradients
        self.optimizer.zero_grad()
        # calculate loss
        loss = self.criterion(target, pred)
        # calculate new gradient
        loss.backward()

        # update model parameters
        self.optimizer.step()