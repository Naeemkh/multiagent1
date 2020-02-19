from domain import *


import random
import matplotlib.animation as animation
mydomain = Domain((10,10))

mywall = []
mystorage = []
mygold=[]

# for i in range(0,12):
#     mywall.append([i,8])

for j in range(4,10):
    mywall.append([3,j])

mydomain.add_wall(mywall)


for i in range(0,8):
    for j in range(0,2): 
        mygold.append([i,j])


for i in range(0,3):
    for j in range(5,10): 
        mystorage.append([i,j])


mydomain.add_gold(mygold)
mydomain.add_storage(mystorage)
mydomain.add_agent([[1,1]])



# Run for individual plots
# actions = ['Up','Down','Left','Right','Pickup','Dropoff']
# for i in range(500):
#     a = random.randint(0,5)
#     reward = mydomain.action(actions[a])
#     mydomain.plot_domain(True)       
#     print(f"Action: {actions[a]}, reward: {reward}")
#     #input("Press Enter to continue...")


## Run for gif animation

fig2 = plt.figure(2, figsize=(6, 6))
#gridspec.GridSpec(1,3)
# plotting domain
#plt.subplot2grid((1,3),(0,0), colspan=2, rowspan=1)
ax = plt.gca()

ax.set_xticks(np.arange(0.5, mydomain.ncols, 1))
ax.set_yticks(np.arange(0.5, mydomain.nrows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])

ims = []
actions = ['Up','Down','Left','Right','Pickup','Dropoff']

for i in range(1000):
    a = random.randint(0,5)
    reward = mydomain.action(actions[a])
    print(f"Action: {actions[a]}, reward: {reward}")
    im = plt.imshow(mydomain.plot_domain(False), interpolation='none')        
    ims.append([im])

ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True)
ani.save('animation.gif', writer='imagemagick', fps=12)
