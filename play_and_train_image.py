import os
import numpy as np
import copy
import pickle

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from keras.models import load_model

from domain import *
from build_model import *
from util import *


class ReplayBuffer():
    """
    class to handle the training sessions.
    see: https://www.youtube.com/watch?v=5fHngyN8Qhw
    """

    def __init__(self,mem_size,domain_shape,learning_rate,gamma,reuse_domain,n_actions, discrete = False):
        self.mem_size = mem_size
        self.domain_shape = domain_shape
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_name = 'model_'+str(domain_shape[0])+'_'+str(domain_shape[1])+'.h5'
        self.reuse_domain = reuse_domain
        self.flatten_domain_size = domain_shape[0]*domain_shape[1]
        self.state_memory = np.zeros((self.mem_size, self.flatten_domain_size))
        self.new_state_memory = np.zeros((self.mem_size, self.flatten_domain_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_counter = 0
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1


    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem,batch_size)

        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        rewards = self.reward_memory[batch]

        return state, actions, rewards, state_, terminal


class PlayAgent(object):

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, 
                      domain_shape, epsilon_dec = 0.99996, epsilon_end=0.01,
                      mem_size = 1000000, fname= 'dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = 'model_'+str(domain_shape[0])+'_'+str(domain_shape[1])+".h5"
        reuse_domain = False

        self.memory = ReplayBuffer(mem_size,domain_shape, alpha, gamma, reuse_domain, n_actions, discrete=True)
        
        bm = BuildModel(domain_shape)
        self.mymodel = bm.get_initial_predition_model()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.mymodel.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values).astype(int)

        q_eval = self.mymodel.predict(state)
        q_next = self.mymodel.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next,axis=1)*done
        

        _ = self.mymodel.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

    def save_model(self):
        self.mymodel.save(self.model_file)

    def load_model(self):
        self.mymodel = load_model(self.model_file)



def save_figure(canvas, domain_shape,tmp_probs,a, ac,fig_num):
            fig = plt.figure(1, figsize=(9, 6))
            gridspec.GridSpec(2,1)
                        
            # plotting domain
            plt.subplot2grid((2,1),(0,0), colspan=1, rowspan=1)
            ax = plt.gca()
     
            ax.set_xticks(np.arange(0.5, domain_shape[1], 1))
            ax.set_yticks(np.arange(0.5, domain_shape[0], 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])            
            
            plt.grid(True)
            plt.imshow(canvas, interpolation=None) 
            
            # plotting parameters
            plt.subplot2grid((2,1),(1,0), colspan=1, rowspan=1)
            plt.axis('off')
            plt.text(0,0.95, 'Prediction Probability: ')
            plt.text(0,0.85, tmp_probs)
            plt.text(0,0.75, f"Chosen action: {ac[a]}")
            

            plt.savefig(f'figure_{str(fig_num)}.pdf')



if __name__ == "__main__":

        domain_shape = (10,10)
        mydomain = create_random_domain(domain_shape,10,5,10)
        mydomain1 = copy.deepcopy(mydomain)

        with open(f'domain_sample_{domain_shape[0]}_{domain_shape[1]}.pkl', 'wb') as outfile:
            pickle.dump(mydomain, outfile, pickle.HIGHEST_PROTOCOL)



        n_games = 1000
        my_agent = PlayAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, domain_shape=domain_shape, n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.01)


        scores = []
        eps_history = []

        figure_number = 1

        for i in range(n_games):
            done = False
            score = 0
            # mydomain = create_random_domain(domain_shape,1,2,1)
            mydomain = copy.deepcopy(mydomain1)

            observation = mydomain.get_state()
            jj = 1
            plotit = True
            while not done:
                action = my_agent.choose_action(observation[0])
                reward, done = mydomain.action(mydomain.actions[action])
                # print(done)
                new_observation = mydomain.get_state()
                score += reward
                my_agent.remember(observation, action, reward,new_observation, done)
                observation = new_observation
                my_agent.learn()
                # print('xx')
                
                if jj % 5 == 0 and i % 50 == 0 and plotit:
                    # print (f"jj: {jj} and i: {i}")
                    current_state = mydomain.get_state()
                    tmp_probs = my_agent.mymodel.predict(current_state)
                    a = np.argmax(tmp_probs[0])
                    save_figure(mydomain.plot_domain(False),domain_shape,tmp_probs,a,mydomain.actions,figure_number)
                    figure_number += 1 
                    plotit = False

                jj += 1
            
            eps_history.append(my_agent.epsilon)
            scores.append(score)
            ave_score = np.mean(scores[max(0,i-100):(i+1)])
            print(f'episode: {i}, epsilon: {my_agent.epsilon}, score: {ave_score}, ')
            
            if i % 100 == 0 and i > 0:
                my_agent.save_model()
                
        plt.figure()
        plt.plot(scores)
        plt.show()












