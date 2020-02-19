import os
import numpy as np

from domain import *
from build_model import *
from util import *

import matplotlib.animation as animation

import pickle
import subprocess
import webbrowser



def save_figure(canvas, domain_shape,fig_num):
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
            # plt.text(0,0.95, 'Prediction Probability: ')
            # plt.text(0,0.85, tmp_probs)
            # plt.text(0,0.75, f"Chosen action: {ac[a]}")
            

            plt.savefig(f'train_figure_{str(fig_num)}.pdf')




domain_shape = (4,4)


## create domain

for ia in range(2):
    
    mydomain = create_random_domain(domain_shape,2,2,2)

        
    fig_num = 1
    mydata = []

    while mydomain.total_golds != 0:
    # while fig_num < 2:

        current_state = mydomain.get_state()
        # save_figure(canvas, domain_shape,tmp_probs,a, ac,fig_num)
        save_figure(mydomain.plot_domain(False),domain_shape,fig_num)
        # subprocess.Popen([f"open train_figure_{fig_num}.pdf"],shell=True)
        os.startfile(f"train_figure_{fig_num}.pdf")
        # webbrowser.open_new(f'train_figure_{fig_num}.pdf')

        user_action = int(input("What action to take? "))
        
       
        mydomain.action(mydomain.actions[user_action])
        target = np.array([-0.5,-0.5,-0.5,-0.5], dtype='float32')
        target[user_action] = 1
        mydata.append([current_state[0],target])
        fig_num += 1
    
    
    
    try:
        with open(f'episode_data_{domain_shape[0]}_{domain_shape[1]}.pkl', 'rb') as infile:
            old = pickle.load(infile)
        
        old.extend(mydata)
    
        with open(f'episode_data_{domain_shape[0]}_{domain_shape[1]}.pkl', 'wb') as outfile:
            pickle.dump(old, outfile, pickle.HIGHEST_PROTOCOL)
    
    except FileNotFoundError:
        with open(f'episode_data_{domain_shape[0]}_{domain_shape[1]}.pkl', 'wb') as outfile:
            pickle.dump(mydata, outfile, pickle.HIGHEST_PROTOCOL)
    
    







