import os
import numpy as np

from domain import *
from build_model import *
from util import *
import pickle

import matplotlib.animation as animation

domain_shape = (10,10)

model_name = 'model_'+str(domain_shape[0])+'_'+str(domain_shape[1])+'.h5'


if os.path.exists(model_name):
   # load existing model
    bm = BuildModel(domain_shape)
    mymodel = bm.get_initial_predition_model()
    print(mymodel.summary()) 
    mymodel.load_weights(model_name)
    print("Weights loaded from disk.")

else:
   # build new model.
    print(f"{model_name} to test is not available.")



## create domain
# mydomain = Domain(domain_shape)
# mywall = []
# mystorage = []
# mygold=[]

# for i in range(3,4):
#     mywall.append([i,4])

# mydomain.add_wall(mywall)

# for i in range(3,4):
#     for j in range(0,3): 
#         mygold.append([i,j])

# # for i in range(0,4):
# #     for j in range(0,4): 
# #         mygold.append([i,j])

# mydomain.add_gold(mygold)

# # for i in range(0,5):
# #     for j in range(3,5): 
# #         mystorage.append([i,j])

# # for i in range(0,3,2):
# #     for j in range(0,6,2): 
# #         mystorage.append([i,j])

# mystorage.append([0,3])

# mydomain.add_storage(mystorage)
# mydomain.add_agent([[0,0]])

# # mydomain.plot_domain(True)



# mydomain = create_random_domain(domain_shape,1,2,1)


with open(f'domain_sample_{domain_shape[0]}_{domain_shape[1]}.pkl', 'rb') as infile:
    mydomain = pickle.load(infile)


fig = plt.figure(1, figsize=(6, 6))
ax = plt.gca()
ax.set_xticks(np.arange(0.5, mydomain.ncols, 1))
ax.set_yticks(np.arange(0.5, mydomain.nrows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ims = []
i = 0

print("++++++++ Here +++++++++")
epsilon = 1.9
# while mydomain.total_golds != 0 or i< 50:
im = plt.imshow(mydomain.plot_domain(False), interpolation=None) 
ims.append([im])

while  i<100:

    if mydomain.total_golds == 0:
        break

    random_number = random.random()
     
    if random_number < epsilon:
        # choose neural net prediction
        # print("Line 146")
        current_state = mydomain.get_state()
        a = np.argmax(mymodel.predict(current_state)[0])
        reward, done = mydomain.action(mydomain.actions[a])

    else:
        # choose random action
        # print("Line 157")
        a = random.randint(0,3)
        # current_state = mydomain.get_state()
        reward = mydomain.action(mydomain.actions[a])


    # current_state = mydomain.get_state()
    # if not current_state:
    #     a = 1
    # a = np.argmax(mymodel.predict(current_state)[0])
    # reward = mydomain.action(mydomain.actions[a])
    print(f"Action: {mydomain.actions[a]}, reward: {reward}")
    im = plt.imshow(mydomain.plot_domain(False), interpolation=None)        
    ims.append([im])
    i += 1

# print(ims)

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
ani.save('animation_test.gif', writer='imagemagick', fps=4)
