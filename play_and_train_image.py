import os
import numpy as np
import copy
import pickle

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from domain import *
from build_model import *
from util import *


 
figure_number = 1
buffer_data = []

learning_rate = 0.005
gamma = 0.8 # discount
domain_shape = (10,10)

reuse_domain = True

try:
    with open(f'episode_data_{domain_shape[0]}_{domain_shape[1]}.pkl', 'rb') as infile:
        episode_data = pickle.load(infile)
except FileNotFoundError:
    episode_data = []


def raw_data_to_training(mydata, prediction_model, learning_rate, gamma):
    """
    s1: Current State
    s2: Next state
    a: action that takes S1 to S2
    R: instant reward in going from S1 to S2 while taking action ac.
    gamma: discount factor. We want to also consider longterm goals.
    Q(s,a): Quality factor being at state s and taking action a.
    Q(s1,a) = R + gamma * max Q(s2,ai)
    """
    current_state, instant_reward, action, next_state = mydata

    # First predict target by a neural network.
    target_cur = prediction_model.predict(current_state)[0]
    target_next = prediction_model.predict(next_state)[0]
    
    # Second modify the Q value of the action you took.   
    tc_old = target_cur[action]

    # uncomment for fine tuning
    target_cur[action] = (1 - learning_rate) * target_cur[action] + learning_rate*(instant_reward + gamma * (max(target_next)))
    
    tc_new = target_cur[action]  
    return [current_state[0],target_cur],[tc_old,tc_new]  

    

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



model_name = 'model_'+str(domain_shape[0])+'_'+str(domain_shape[1])


if os.path.exists(model_name+".h5"):
    # load existing model
    bm = BuildModel(domain_shape)
    mymodel = bm.get_initial_predition_model()
    mymodel.load_weights(model_name+".h5")
    print("Weights loaded from disk.")
    
else:
    # build new model.
     bm = BuildModel(domain_shape)
     mymodel = bm.get_initial_predition_model()
     print("Weights not found. Creating a model with random weights.")
    


if reuse_domain:
    with open(f'domain_sample_{domain_shape[0]}_{domain_shape[1]}.pkl', 'rb') as infile:
        mydomain = pickle.load(infile)

else:
    mydomain_org = create_random_domain(domain_shape,10,5,1)
    mydomain1 = copy.deepcopy(mydomain_org)
    with open(f'domain_sample_{domain_shape[0]}_{domain_shape[1]}.pkl', 'wb') as outfile:
      pickle.dump(mydomain_org, outfile, pickle.HIGHEST_PROTOCOL)


# episode is the whole start to end process. 
# Here I meant to say going from one state to another. 
# The name should be changed. 
max_num_data_episode = 5000  
max_num_data_buffer = 100

for ii in range(400000):
    #TODO: come up with better control to end the for loop

    # In the initial iteration, it performs better with random actions.
    # With higher random action it has more discovery power.     
    if ii < 100:
        epsilon = 0.1
    elif ii < 200:
        epsilon = 0.5
    elif ii < 500:
        epsilon = 0.7
    else:
        epsilon = 0.9


    with open(f'domain_sample_{domain_shape[0]}_{domain_shape[1]}.pkl', 'rb') as infile:
        mydomain = pickle.load(infile)

    ### Uncomment to see the image of generated domain
    #mydomain.plot_domain(True)
      

    episode_number = 1
    k = 0
    ims = []
    process = []

    while episode_number < 100:
        
        if ii % 800 == 0 and episode_number == 8:

            current_state = mydomain.get_state()
            tmp_probs = mymodel.predict(current_state)
            a = np.argmax(tmp_probs[0])
            save_figure(mydomain.plot_domain(False),domain_shape,tmp_probs,a,mydomain.actions,figure_number)
            figure_number += 1

        random_number = random.random()
    
        if random_number < epsilon:
            # choose action based on deep neural net prediction
            current_state = mydomain.get_state()
            a = np.argmax(mymodel.predict(current_state)[0])
            reward = mydomain.action(mydomain.actions[a])
            next_state = mydomain.get_state()
            episode = [current_state, reward, a, next_state]
            tr_data,tc = raw_data_to_training(episode,mymodel,learning_rate, gamma)
            buffer_data.append(tr_data)

    
        else:
            # choose random action
            a = random.randint(0,3)
            current_state = mydomain.get_state()
            reward = mydomain.action(mydomain.actions[a])
            next_state = mydomain.get_state()
            episode = [current_state, reward, a, next_state]
            tr_data,tc = raw_data_to_training(episode,mymodel,learning_rate, gamma)
            buffer_data.append(tr_data)

        
        if len(buffer_data) >= max_num_data_buffer:
    
            if len(episode_data) >= max_num_data_episode:
                del episode_data[0:max_num_data_buffer]
                episode_data.extend(buffer_data)
                buffer_data.clear()
            else:
                episode_data.extend(buffer_data)
                buffer_data.clear()
    
        else:
            pass
    
        if episode_data and ii % 100 == 0 and episode_number == 1:
            
            input_1 = episode_data[0][0]
            label_1 = episode_data[0][1]
            
            for i in range(1,len(episode_data)):
                input_1 = np.vstack([input_1,episode_data[i][0]])
                label_1 = np.vstack([label_1,episode_data[i][1]])
    
            mymodel.fit(input_1,label_1, epochs=20, batch_size=256, verbose=1)
            print("=============== Model is reTrained ==================")
    

        if ii % 500 == 0 and episode_number==1:
            h5file = model_name + ".h5"
            json_file = model_name + ".json"
            mymodel.save_weights(h5file, overwrite=True)
            model_json = mymodel.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            print("Model weights are saved.")

            
        if mydomain.total_golds == 0:
            print("All golds are carried into the storage.")
            break

        episode_number += 1
    

# Save trained model weights and architecture, this will be used by the visualization code
h5file = model_name + ".h5"
json_file = model_name + ".json"
mymodel.save_weights(h5file, overwrite=True)
model_json = mymodel.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Done!")








