import torch
import numpy as np 
import pandas as pd
import os
from statistics import mean

if os.path.exists('agent_log.csv'):
    os.remove('agent_log.csv')

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

            '''
            last_next_state = self.buffer.states[-1]  # Get the last state in the buffer
            with torch.no_grad():
                next_value = self.policy.critic(last_next_state).squeeze()  # state AFTER buffer ends

            for i in reversed(range(len(self.buffer.rewards))):
                next_val = next_value if i == len(self.buffer.rewards) - 1 else old_state_values[i + 1]
                delta = self.buffer.rewards[i] + GAMMA * next_val - old_state_values[i]
                gae = delta + GAMMA * LAMBDA * gae
                advantages[i] = gae
            '''
            
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + GAMMA * (0 if i == len(rewards)-1 else old_state_values[i + 1]) - old_state_values[i]
                gae = delta + GAMMA * LAMBDA * gae
                advantages[i] = gae
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
 


class RLAgent:
    def __init__(self):
        K_epochs = 10              
        eps_clip = 0.2            
        gamma = 0.95                
        lr_actor = 0.0001 
        lr_critic = 0.0001
        hidden_neurons = 128
        
        self.max_vis_sat = 20
        self.num_features = 2
        state_dim = self.max_vis_sat * self.num_features
        self.ppo = PPO(state_dim, self.max_vis_sat, hidden_neurons, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
        self.counter = 1
        self.ue_ids = []
        self.states = []
        self.actions = []
        self.rewards = []

    def rl_algorithm_selection(self, ue_id, sat_infos, states):
        self.ue_ids.append(ue_id)
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

        self.ue_ids = []
        self.actions = []
        self.rewards = []

        
        return
    
    def save_model(self, path = "checkpoints/agent.pt"):
        self.ppo.save(path)

    def load_model(self, path = "checkpoints/agent.pt"):
        if not os.path.isfile(path):
            return None
        self.ppo.load(path)


