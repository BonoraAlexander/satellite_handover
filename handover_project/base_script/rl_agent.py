import torch
import numpy as np 
import pandas as pd
import os
from statistics import mean
from abc import ABC
from torch.nn import Module
import random
import math

# if os.path.exists('agent_log.csv'):
#    os.remove('agent_log.csv')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        # del self.actions[:]
        # del self.states[:]
        # del self.logprobs[:]
        # del self.rewards[:]
        # del self.state_values[:]
        # del self.is_terminals[:]
        self.actions = self.actions[-(len(self.actions)-512):] 
        self.states = self.states[-(len(self.states)-512):]
        self.logprobs = self.logprobs[-(len(self.logprobs)-512):]
        self.rewards = self.rewards[-(len(self.rewards)-512):]
        self.state_values = self.state_values[-(len(self.state_values)-512):]
        self.is_terminals = self.is_terminals[-(len(self.is_terminals)-512):]


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_neurons, device):
        super(ActorCritic, self).__init__()
        self.device = device

        self.actor = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_neurons),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_neurons, hidden_neurons),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_neurons, action_dim),
                        torch.nn.Softmax(dim=-1)
                    )
        
        # critic
        self.critic = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_neurons),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_neurons, hidden_neurons),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_neurons, 1)
                    )
        
    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        action_probs = self.actor(state) 
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy



class PPO:
    def __init__(self, state_dim, action_dim, hidden_neurons, lr_actor, lr_critic, gamma, K_epochs, eps_clip, update_timestep=512, batch_size=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.learning_steps = 0
        self.update_timestep = update_timestep
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, hidden_neurons, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_neurons, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = torch.nn.MSELoss()


    def select_action(self, state):

        self.learning_steps += 1

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.tolist()


    def update(self):
            
        # Monte Carlo estimate of returns
        

        #Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # # TD(0) estimation of returns
        # rewards = []
        # next_state_value = 0  # Assume the value of a terminal state is 0
        # for reward, is_terminal, state_value in zip(reversed(self.buffer.rewards), 
        #                                             reversed(self.buffer.is_terminals), 
        #                                             reversed(self.buffer.state_values)):
        #     if is_terminal:
        #         next_state_value = 0  # Terminal state
        #     # TD(0) update for return
        #     next_state_value = reward + (self.gamma * next_state_value)
        #     rewards.insert(0, next_state_value)

        # # Convert rewards to tensor and normalize
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states[:self.update_timestep], dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:self.update_timestep], dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:self.update_timestep], dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:self.update_timestep], dim=0)).detach().to(self.device)

        rewards = self.buffer.rewards[:self.update_timestep]
        with torch.no_grad():
            advantages = torch.zeros(len(rewards)).to(self.device)
            gae = 0
            GAMMA = 0.95
            LAMBDA = 0.95

            last_next_state = self.buffer.states[-1]  
            with torch.no_grad():
                next_value = self.policy.critic(last_next_state).squeeze() 
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + GAMMA * (next_value if i == len(rewards)-1 else old_state_values[i + 1]) - old_state_values[i]
                advantages[i] = delta + GAMMA * LAMBDA * gae
            returns = advantages + old_state_values

          
        for _ in range(self.K_epochs):
            num_samples = len(rewards)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                mini_batch_indices = indices[start:end]

                mb_old_states = old_states[mini_batch_indices]
                mb_old_actions = old_actions[mini_batch_indices]
                mb_old_logprobs = old_logprobs[mini_batch_indices]
                mb_old_state_values = old_state_values[mini_batch_indices]
                mb_advantages = advantages[mini_batch_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_returns = returns[mini_batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_old_states, mb_old_actions)

                state_values = torch.squeeze(state_values)
                
                ratios = torch.exp(logprobs - mb_old_logprobs.detach())

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_advantages

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mb_returns.detach()) - 0.01 * dist_entropy
                
                # self.entropies.append(dist_entropy.cpu().detach().numpy().mean())
                # self.losses.append(loss.cpu().detach().numpy().mean())
                # self.values.append(state_values.cpu().detach().numpy().mean())
                #self.rewards.append(rewards.cpu().detach().numpy().mean())

                self.optimizer.zero_grad()
                loss.mean().backward()
                #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, path):

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
 
        checkpoint = {
            "policy_state_dict":     self.policy.state_dict(),
            "policy_old_state_dict": self.policy_old.state_dict(),
            "optimizer_state_dict":  self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
 
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
 
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_old_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
 
class LinearNeuralNetwork(Module, ABC):
    """
    It defines a basic linear neural network to estimate the action q-values
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):

        super(LinearNeuralNetwork, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, 128)
        self.batch_norm_1 = torch.nn.BatchNorm1d(128)
        self.linear_2 = torch.nn.Linear(128, 256)
        self.batch_norm_2 = torch.nn.BatchNorm1d(256)
        self.linear_3 = torch.nn.Linear(256, 256)
        self.batch_norm_3 = torch.nn.BatchNorm1d(256)
        self.linear_4 = torch.nn.Linear(256, 128)
        self.batch_norm_4 = torch.nn.BatchNorm1d(128)
        self.linear_5 = torch.nn.Linear(128, output_dim)

        torch.nn.init.kaiming_uniform_(self.linear_1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear_2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear_3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear_4.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear_5.weight, nonlinearity='relu')

        torch.nn.init.zeros_(self.linear_1.bias)
        torch.nn.init.zeros_(self.linear_2.bias)
        torch.nn.init.zeros_(self.linear_3.bias)
        torch.nn.init.zeros_(self.linear_4.bias)
        torch.nn.init.zeros_(self.linear_5.bias)

    def forward(self, x: torch.Tensor):
        """
        Compute the q values of the input tensor x
        """

        x = torch.nn.functional.relu(self.linear_1(x))
        x = self.batch_norm_1(x)
        x = torch.nn.functional.relu(self.linear_2(x))
        x = self.batch_norm_2(x)
        x = torch.nn.functional.relu(self.linear_3(x))
        x = self.batch_norm_3(x)
        x = torch.nn.functional.relu(self.linear_4(x))
        x = self.batch_norm_4(x)
        return self.linear_5(x)
    '''
    def __init__(self,
                 input_dim,
                 output_dim,
                 ):

        super(LinearNeuralNetwork, self).__init__()
        self.batch_norm_0 = torch.nn.BatchNorm1d(input_dim)
        self.linear_1 = torch.nn.Linear(input_dim, 128)
        self.batch_norm_1 = torch.nn.BatchNorm1d(128)
        self.linear_2 = torch.nn.Linear(128, 128)
        self.batch_norm_2 = torch.nn.BatchNorm1d(128)
        self.linear_3 = torch.nn.Linear(128, output_dim)
        #self.V = torch.nn.Linear(16, 1)
        #self.A = torch.nn.Linear(16, output_dim)

        torch.nn.init.kaiming_uniform_(self.linear_1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear_2.weight, nonlinearity='relu')
        torch.nn.init.uniform_(self.linear_3.weight)

        torch.nn.init.zeros_(self.linear_1.bias)
        torch.nn.init.zeros_(self.linear_2.bias)
        torch.nn.init.uniform_(self.linear_3.bias)

    def forward(self, x: torch.Tensor):
        """
        Compute the q values of the input tensor x
        """
        x = self.batch_norm_0(x)
        x = torch.nn.functional.relu(self.linear_1(x))
        x = self.batch_norm_1(x)
        x = torch.nn.functional.relu(self.linear_2(x))
        x = self.batch_norm_2(x)
        #V = self.V(x)
        #A = self.A(x)
        #Q = V + (A - A.mean(dim=1, keepdim=True))
        return self.linear_3(x)
    '''

class DDQL(object):
    """
    class DQL
    It implements the Double Q Learning algorithm

    """

    def __init__(self,
                 primary_net,
                 target_net,
                 optimizer,
                 gamma,
                 batch_size,
                 target_replace,
                 memory_capacity):

        # Target and evaluation networks used in the training

        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__primary_net: LinearNeuralNetwork = primary_net  # Choose new actions
        self.__target_net: LinearNeuralNetwork = target_net  # Estimate the future q values
        self.__primary_net.to(self.__device)
        self.__target_net.to(self.__device)

        # Learning weights optimizer
        self.__optimizer = optimizer

        # Counters of the number of learning steps and the number of transition inserted in the memory replay
        self.__learn_step = 0
        self.__memory_step = 0

        # Learning parameters

        self.__batch_size = batch_size
        self.__target_replace = target_replace
        self.__gamma = gamma  # Should be close to 1

        # Memory replay

        self.__memory_capacity = memory_capacity
        self.__memory = []
        self._step = 0


    def store_transition(self, state, action, reward, new_state):

        """
        Insert a new transition (state, action, reward, new state) in the memory replay
        """

        if self.__memory_step < self.__memory_capacity:
            self.__memory.append([state, action, reward, new_state])  # Transition
        else:
            index = int(self.__memory_step % self.__memory_capacity)  # Index of the transition
            self.__memory[index] = [state, action, reward, new_state]  # Transition
        self.__memory_step += 1  # Increase the number of total transition

    def ready(self):
        return self.__memory_step >= self.__memory_capacity#3 * self.__batch_size

    def step(self):
        """
        Take a batch from the memory replay and perform a training step
        """
        #self.__primary_net.half()
        #self.__target_net.half()
        batch_transitions = random.sample(self.__memory, self.__batch_size)

        batch_states = torch.stack(
            [torch.tensor(transition[0], dtype=torch.float32) for transition in batch_transitions]).to(self.__device)
        batch_actions = torch.tensor(
            [transition[1] for transition in batch_transitions], dtype=torch.int64).view(-1, 1).to(self.__device)
        batch_rewards = torch.tensor(
            [transition[2] for transition in batch_transitions], dtype=torch.float16).to(self.__device)
        batch_new_states = torch.stack(
            [torch.tensor(transition[3], dtype=torch.float32) for transition in batch_transitions]).to(self.__device)

        actual_q_values = self.__primary_net.forward(batch_states).gather(1, batch_actions).squeeze(1)

        with torch.no_grad():
            next_q_values: torch.FloatTensor = self.__primary_net.forward(batch_new_states)
            next_actions = torch.argmax(next_q_values, dim=1).view(-1, 1)
            target_q_values = batch_rewards + self.__gamma * \
                              self.__target_net.forward(batch_new_states).gather(1, next_actions).squeeze(
                                  1)

        losses = (actual_q_values - target_q_values) ** 2
        # Compute the mean loss of the batch
        loss_mean = losses.mean()

        # Optimization ddql
        self.__optimizer.zero_grad()
        loss_mean.backward()
        self.__optimizer.step()

        self.__learn_step += 1

        # Periodically, replace the weights of the target network with those of the primary network
        if self.__learn_step % self.__target_replace == 0:
            self.__target_net.load_state_dict(self.__primary_net.state_dict())
        
        loss_mean = loss_mean.item()

        return loss_mean
    
    def get_device(self):
        return self.__device
    
    def get_primary_net(self):
        return self.__primary_net
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "primary_net_state_dict": self.__primary_net.state_dict(),
            "target_net_state_dict":  self.__target_net.state_dict(),
            "optimizer_state_dict":   self.__optimizer.state_dict(),
            "learn_step":             self.__learn_step,
            "memory_step":            self.__memory_step,
            "step":                   self._step,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.__device)

        self.__primary_net.load_state_dict(checkpoint["primary_net_state_dict"])
        self.__target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.__learn_step = checkpoint.get("learn_step", 0)
        self.__memory_step = checkpoint.get("memory_step", 0)
        self._step = checkpoint.get("step", 0)

        for state in self.__optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.__device)
class PPOAgent:
    def __init__(self):
        K_epochs = 10              
        eps_clip = 0.2            
        gamma = 0.95                
        lr_actor = 0.0001 
        lr_critic = 0.0001
        hidden_neurons = 128
        
        self.max_vis_sat = 30
        self.num_features = 3
        state_dim = self.max_vis_sat * self.num_features
        self.ppo = PPO(state_dim, self.max_vis_sat, hidden_neurons, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
        self.counter = 1
        self.states = []
        self.actions = []
        self.rewards = []

    def rl_algorithm_selection(self, ue_id, sat_infos, states):
        # Pad the states to have a fixed size of max_vis_sat
        if len(states) < self.max_vis_sat:
            for _ in range(self.max_vis_sat - len(states)):
                states.append([0]*self.num_features)
        
        states = np.asarray(states).reshape(1, -1)
        self.states.append(states)
        action = self.ppo.select_action(states)[0]
        self.actions.append(action)
        
        if action < len(sat_infos):
            return sat_infos[action]
        
        # If the action is greater than the number of available satellites, select an invalid satellite (None) -> out of service
        else:
            invalid_sat_info = list(sat_infos[0])
            invalid_sat_info[0] = None
            return tuple(invalid_sat_info)
        

    def rl_rewards(self, rewards):
        for reward in rewards:
            self.ppo.buffer.rewards.append(reward[1])
            self.ppo.buffer.is_terminals.append(False)
            self.rewards.append(reward[1])

        if self.ppo.learning_steps and (self.ppo.learning_steps >= self.counter*self.ppo.update_timestep):
            self.ppo.update()
            self.counter += 1

        log = pd.DataFrame([{
            'actions': self.actions,
            'reward': mean(self.rewards),
        }])
                
        write_header = not os.path.exists('agent_log.csv')
        log.to_csv('agent_log.csv', index=False, mode='a', header=write_header)

        self.actions = []
        self.rewards = []

        
        return
    
    def save_model(self, path = "checkpoints/agent.pt"):
        self.ppo.save(path)

    def load_model(self, path = "checkpoints/agent.pt"):
        if not os.path.isfile(path):
            return None
        self.ppo.load(path)




class DDQLAgent():
    """
    class DQLAgent
    It implements the Double Q Learning algorithm
    """

    def __init__(self,nn_params = None,
                 ):
        
        self.max_vis_sat = 30
        self.num_features = 3
        state_dim = self.max_vis_sat * self.num_features
        action_dim = self.max_vis_sat
        gamma = 0.95
        batch_size = 32
        target_replace = 1000
        memory_capacity = 1000
        learning_rate = 0.0001
        eps = 0.01
        weight_decay = 0.0001

        self._state_dim = state_dim
        self._action_dim = action_dim
        primary_net = LinearNeuralNetwork(state_dim, action_dim)
        target_net = LinearNeuralNetwork(state_dim, action_dim)
        target_net.load_state_dict(primary_net.state_dict())
        optimizer = torch.optim.Adam(primary_net.parameters(),
                                     lr=learning_rate, eps=eps, weight_decay=weight_decay)

        if nn_params is not None:
            primary_net.load_state_dict(nn_params)
            target_net.load_state_dict(nn_params)

        self.__ddql = DDQL(primary_net, target_net, optimizer, gamma, batch_size, target_replace, memory_capacity)

        self._states = []
        self._actions = []
        self._rewards = []

        self.actions = []
        self.rewards = []

    def get_epsilon(self, step):
        EPS_MIN = 0.1
        EPS_MAX = 1
        LAMBDA = 0.001
        return EPS_MIN + math.exp(-LAMBDA*step) * (EPS_MAX-EPS_MIN)

    def get_action(self, state):
        """
        Choose an action according to the policy
        """
        self.__ddql.get_primary_net().eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.__ddql.get_device())
        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            q_values = self.__ddql.get_primary_net().forward(state_tensor)

        action_chosen = torch.argmax(q_values).item()
        action_performed = action_chosen
        if random.random() < self.get_epsilon(self.__ddql._step):
            action_performed = random.randint(0, self._action_dim - 1)

        self._actions.append(action_performed)
        self.__ddql._step += 1
        return action_performed

    def learn(self):
        """
        Learn from the experience
        """
        loss = None
        # need at least two recorded states to have both s_t and s_{t+1}
        while len(self._states) >= 2 and self._actions and self._rewards:
            s_t = self._states.pop(0)
            a_t = self._actions.pop(0)
            r_t = self._rewards.pop(0)
            s_t1 = self._states[0]  

            self.store_transition(s_t, a_t, r_t, s_t1)

            if self.__ddql.ready():
                self.__ddql.get_primary_net().train()
                loss = self.__ddql.step()

        return loss
    
    def reset(self):
        """Delete all temporary stored data
        """
        self._states = []
        self._actions = []
        self._rewards = []

    def observe_state(self, state):
        """Observe the current state

        Args:
            state (State): state of the user
        """
        self._states.append(state)

    def observe_reward(self, reward):
        """Observe the reward

        Args:
            reward (float): reward received
        """
        self._rewards.append(reward)


    def store_transition(self, state, action, reward, new_state):
        """
        Insert a new transition (state, action, reward, new state) in the memory replay
        """
        self.__ddql.store_transition(state, action, reward, new_state)

    
    def rl_algorithm_selection(self, ue_id, sat_infos, states):
        # Pad the states to have a fixed size of max_vis_sat
        if len(states) < self.max_vis_sat:
            for _ in range(self.max_vis_sat - len(states)):
                states.append([0]*self.num_features)
        
        states = np.asarray(states).reshape(-1)
        self.observe_state(states)
        action = self.get_action(states)
        self.actions.append(action)
        
        if action < len(sat_infos):
            return sat_infos[action]
        
        # If the action is greater than the number of available satellites, select an invalid satellite (None) -> out of service
        else:
            invalid_sat_info = list(sat_infos[0])
            invalid_sat_info[0] = None
            return tuple(invalid_sat_info)
        

    def rl_rewards(self, rewards):
        for reward in rewards:
            self.observe_reward(reward[1])
            self.rewards.append(reward[1])  

        self.learn()

        log = pd.DataFrame([{
            'actions': self.actions,
            'reward': mean(self.rewards),
        }])
                
        write_header = not os.path.exists('agent_log.csv')
        log.to_csv('agent_log.csv', index=False, mode='a', header=write_header)

        self.actions = []
        self.rewards = []

        
        return
    
    def save_model(self, path = "checkpoints/agent.pt"):
        return
        self.ppo.save(path)

    def load_model(self, path = "checkpoints/agent.pt"):
        return
        if not os.path.isfile(path):
            return None
        self.ppo.load(path)