
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Agent:
    def __init__(self,agent_loc):
        self.has_box = False
        self.color = None
        self.row, self.col = agent_loc
  
    def take_action(self, item):
        """
        An agent takes action or at least tries to do so. 
        The domain will decide whether it will implement 
        the requested action or not. This is important because
        the domain will punish the action that is not valid. 
        As a result, the agent, after some time of learning, 
        should not even try to request an invalid action.  
        In other words, we want to give the agent enough 
        latitude to do whatever it wants.
        """
        arow = self.row
        acol = self.col
        
        pickup = False
        dropoff = False

        if item == "Up":
            arow -= 1
        elif item == "Down":
            arow += 1
        elif item == "Right":
            acol += 1
        elif item == "Left":
            acol -= 1
        elif item == "Pickup":
            pickup = True
        elif item == "Dropoff":
            dropoff = True
        else:
            print("Action is not defined.")

        return (arow, acol, pickup, dropoff)

    def agent_code_color_update(self):
        if self.has_box:
            self.code  = 7
            self.color = [7,140,43]
        else:
            self.code = 9
            self.color = [235,7,7]


class Cube:
    cube_values = {
        "wall" : [0, [0,0,0]],
        "fspace" : [1, [255,255,255]],
        "gold" : [2,[255,196,0]],
        "storage": [3,[255,224,224]],
        "gold_delivered_storage": [4, [0, 88, 161]]
    }

    def __init__(self):
        self.update("fspace")

    def update(self,ctype):

            self.ctype = ctype
            self.code = self.cube_values[ctype][0]
            self.color = self.cube_values[ctype][1]
   
    def is_available(self):
        if self.ctype == "wall":
            return False
        else:
            return True

    def get_color(self,agent):
        if not agent:
            return self.color
        else:
            agent.agent_code_color_update()
            return agent.color

    def get_code(self,agent):
        if not agent:
            return self.code
        else:
            return agent.code


class Domain:

    def __init__(self, dshape):
        """
        shape: tuple: (v,h): number of blocks in horizontal and vertical direction
        """
        self.nrows, self.ncols = dshape
        self.initiat_domain()
        self.color_space_holder = np.zeros((self.nrows,self.ncols,3))
        self.code_space_holder = np.zeros((self.nrows,self.ncols))
        self.can_add_agent = True
      
       
    #    list of possibilities for reward:
    #    hitting_wall (n_hw) - 
    #    hitting_border (n_hb) -
    #    hittin_gold_delivered_storage (n_hgds) -
    #    enter_cube_with_gold_while_has_gold (n_ecwg_hg) -
    #    enter_cube_with_gold_while_hasnot_gold (p_ecwg_hng) +
    #    attemp_to_pickup_gold_at_fspace (n_atpg_afs) -
    #    attemp_to_pickup_gold_at_gold_cube_has_gold (n_atpg_agc_hg) -
    #    attemp_to_pickup_gold_at_gold_cube_hasnot_gold (p_atpg_agc_hng) +
    #    wandering_around (n_wa) -
    #    enter_storage_while_has_gold (p_es_hg) +
    #    enter_storage_while_hasnot_gold (n_es_hng) -
    #    attemp_to_drop_gold_at_storage_agent_has_gold (p_atdg_as_ahg) +
    #    attemp_to_drop_gold_at_storage_agent_hasnot_gold (n_atdg_as_ahng) -
    #    attemp_to_drop_gold_at_gold_delivered_storage (n_atdg_agds) -
    #    attemp_to_drop_gold_at_cube_with_gold (n_atdg_acwg) -
    #    attemp_to_dorp_gold_has_not_have_one (n_atdg_hng) -
    #    attemp_to_drop_gold_at_free_space (n_atdg_fspace) -
       

        self.rewards = {'n_hw': -0.8,
                        'n_hb': -0.5,
                        'n_hgds': -0.3,
                        'n_ecwg_hg': -0.3,
                        'p_ecwg_hng': +0.6,
                        'n_atpg_afs': -0.2,
                        'n_atpg_agc_hg': -0.6,
                        'p_atpg_agc_hng': +0.7,
                        'n_wa': -0.02,
                        'p_es_hg': +0.6,
                        'n_es_hng': -0.4,
                        'p_atdg_as_ahg': +1.0,
                        'n_atdg_as_ahng': -0.2,
                        'n_atdg_agds': -0.5,
                        'n_atdg_hng': -0.02,
                        'n_atdg_acwg': -0.02,
                        'n_atdg_fspace': -0.2
                         }
        
    def initiat_domain(self):
        "Initialize the domain as n*n matri, 1 is free space"
        self.domain_mat = np.array([[Cube() for i in range(self.ncols)] for j in range(self.nrows)])

    def upade_state(self,loc,status):
        """ loc: location of the agent.
            status: what will be.
        """
        self.domain_mat[loc[0]][loc[1]].update(status)

    def compute_reward(self):
        pass

    def action(self,action_item):
        """
        There are 6 actions:
        - Move left 
        - Move right
        - Move up
        - Move down
        - Pickup the box
        - Dropoff the box
        """
        
        if self.can_add_agent:
            print("There is no agent to take any action")
        else:
            # print(f"Agent row: {self.agent.row}, col: {self.agent.col}")
            
            # At this step, agent shows what it wants to do, e.g., wants to get out of the domain.
            # However, Domain decides whether it is a valid move and give rewards accordingly. 
            (new_row, new_col, pickup, dropoff) = self.agent.take_action(action_item)


            if pickup:
                if self.domain_mat[self.agent.row][self.agent.col].ctype == "gold":
                    # currently there is a gold in the place that the agent is located.
                    if not self.agent.has_box:
                        # agent does not have box(gold) so it can pick it up and will get + reward.
                        self.agent.has_box = True
                        self.domain_mat[self.agent.row][self.agent.col].update("fspace")
                        this_action_reward = self.rewards["p_atpg_agc_hng"]
                    else: 
                        # agent has a box, however, tries to get another one. 
                        # Nothing happen, but will recieve - reward.
                        this_action_reward = self.rewards["n_atpg_agc_hg"]
                else:
                    # Agent is at some place that is a free space, however, tries to pickup gold, 
                    # will recieve - rewards.
                    this_action_reward = self.rewards["n_atpg_afs"]
                
                return this_action_reward


            if dropoff:
                if not self.agent.has_box:
                    # agent tries to drop a box, where as does not have one. 
                    # nothing will happen but will recieve - reward.
                    this_action_reward = self.rewards["n_atdg_hng"]
                else:
                    # The agent has the gold.
                    if self.domain_mat[self.agent.row][self.agent.col].ctype == "storage":
                        # the storage is ready to recieve the gold.
                        self.agent.has_box = False
                        self.domain_mat[self.agent.row][self.agent.col].update("gold_delivered_storage")
                        this_action_reward = self.rewards["p_atdg_as_ahg"]

                    elif self.domain_mat[self.agent.row][self.agent.col].ctype == "gold_delivered_storage":
                        # The location is not available 
                        # Agent should never get here 
                        print("Bug to fix: Agent should never gets here. Block it")

                    elif self.domain_mat[self.agent.row][self.agent.col].ctype == "gold":
                        # The agent has gold, however, wants to drop it at a cube with gold
                        # nothing will happen, however, will recieve - reward.
                        this_action_reward = self.rewards["n_atdg_acwg"]

                    elif self.domain_mat[self.agent.row][self.agent.col].ctype == "fspace":
                        # The agent has a gold, however, wants to drop it at some free space.
                        # Can do it, but not encouraged--will recieve - reward.
                        self.agent.has_box = False
                        self.domain_mat[self.agent.row][self.agent.col].update("gold")
                        this_action_reward = self.rewards["n_atdg_fspace"]
                    else:
                        print("Bug to fix: This condition is not predicted.")

                return this_action_reward


            if (not pickup) and (not dropoff):
                # the agent wants to change location
                if (new_row < 0 or new_row > self.nrows-1) or (new_col < 0 or new_col > self.ncols-1):
                    # the agent wants to go out of the domain. Nothing will happen,
                    # however, the agent will get - reward.
                    this_action_reward = self.rewards["n_hb"]

                elif self.domain_mat[new_row][new_col].ctype == "wall":
                    # the agent wants to go into the wall. Nothing will happen, 
                    # however, the agent will get - reward.
                    this_action_reward = self.rewards["n_hw"]

                elif self.domain_mat[new_row][new_col].ctype == "gold_delivered_storage":
                    # the agent wants to go into the storage that is gold delivered. 
                    # Nothing will happen, however, the agent will receive - reward.
                    this_action_reward = self.rewards["n_hgds"]

                elif self.domain_mat[new_row][new_col].ctype == "gold":
                    # the agent wants to go into a cube with gold
                    if self.agent.has_box:
                        # already has the gold, should not do this. 
                        # will recieve negative reward.
                        self.agent.row = new_row
                        self.agent.col = new_col
                        this_action_reward = self.rewards["n_ecwg_hg"]
                    else:
                        # has not a gold, will recieve + reward
                        self.agent.row = new_row
                        self.agent.col = new_col
                        this_action_reward = self.rewards["p_ecwg_hng"]
                
                elif self.domain_mat[new_row][new_col].ctype == "fspace":
                    # the agent wandering around, it is ok, however,
                    # will recieve some negative rewards.
                    self.agent.row = new_row
                    self.agent.col = new_col
                    this_action_reward = self.rewards["n_wa"]

                elif self.domain_mat[new_row][new_col].ctype == "storage":
                    # the agent wandering around, it is ok, however,
                    # will recieve some negative rewards.
                    if self.agent.has_box:
                        self.agent.row = new_row
                        self.agent.col = new_col
                        this_action_reward = self.rewards["p_es_hg"]
                    else:
                        self.agent.row = new_row
                        self.agent.col = new_col
                        this_action_reward = self.rewards["n_es_hng"]

                else:
                    print("Bug to fix: This condition is not predicted.")
                    this_action_reward = 0

                return this_action_reward

            

    def add_wall(self,wall_xy):
        for loc in wall_xy:
            if loc[0] < self.nrows and loc[1] < self.ncols:
                self.domain_mat[loc[0]][loc[1]].update("wall")
        
   
    def add_agent(self,agent_loc):

        for loc in agent_loc:

            if (loc[0] < self.nrows and loc[1] < self.ncols) and self.domain_mat[loc[0]][loc[1]].is_available():
                self.agent = Agent(loc)
                self.can_add_agent = False # limit to one agent for now. 


    def add_gold(self,gold_loc):
        for loc in gold_loc:
            if (loc[0] < self.nrows and loc[1] < self.ncols) and self.domain_mat[loc[0]][loc[1]].is_available():
                self.domain_mat[loc[0]][loc[1]].update("gold")

    def add_storage(self,storage_loc):
        for loc in storage_loc:
            if (loc[0] < self.nrows and loc[1] < self.ncols) and self.domain_mat[loc[0]][loc[1]].is_available():
                self.domain_mat[loc[0]][loc[1]].update("storage")
        


    def compute_color(self):
        """color matrix will be used to presetnation purposes."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.agent.row == i and self.agent.col == j:
                    
                     r,g,b = self.domain_mat[i][j].get_color(self.agent)
                   
                else:

                     r,g,b = self.domain_mat[i][j].get_color(None)
                
                self.color_space_holder[i][j][0] = r/255
                self.color_space_holder[i][j][1] = g/255
                self.color_space_holder[i][j][2] = b/255
        

    def comput_code(self):
        """code matrix will be used as an input for neural network"""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.agent.row == i and self.agent.col == j:                    
                     val = self.domain_mat[i][j].get_code(self.agent)                   
                else:
                     val = self.domain_mat[i][j].get_code(None)
                
                self.code_space_holder[i][j] =  val
  

    def plot_domain(self,individual=True):
        
        self.compute_color()
        canvas = np.copy(self.color_space_holder)
        
        if individual:
            fig = plt.figure(1, figsize=(9, 6))
            gridspec.GridSpec(1,3)
                        
            # plotting domain
            plt.subplot2grid((1,3),(0,0), colspan=2, rowspan=1)
            ax = plt.gca()
     
            ax.set_xticks(np.arange(0.5, self.ncols, 1))
            ax.set_yticks(np.arange(0.5, self.nrows, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])            
            
            plt.grid(True)
            plt.imshow(canvas, interpolation='none') 
            
            # plotting parameters
            plt.subplot2grid((1,3),(0,2), colspan=1, rowspan=1)
            plt.axis('off')
            plt.text(0,0.95, "Param1 goes here")
            plt.text(0,0.85, "Param2 goes here")
            plt.text(0,0.75, "Param3 goes here")
            plt.show()
        else:
            return canvas



if __name__=="__main__":
    import random
    import matplotlib.animation as animation
    mydomain = Domain((25,25))

    mywall = []
    mystorage = []
    mygold=[]

    for i in range(0,12):
        mywall.append([i,8])

    for j in range(0,12):
        mywall.append([20,j])
     

    mydomain.add_wall(mywall)


    for i in range(6,10):
        for j in range(0,5): 
            mygold.append([i,j])
    
    for i in range(6,10):
        for j in range(14,22): 
            mygold.append([i,j])

    for i in range(16,20):
        for j in range(1,24): 
            mystorage.append([i,j])

    for i in range(13,15):
        for j in range(2,22): 
            mygold.append([i,j])

   
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

    for i in range(4000):
        a = random.randint(0,5)
        reward = mydomain.action(actions[a])
        print(f"Action: {actions[a]}, reward: {reward}")
        im = plt.imshow(mydomain.plot_domain(False), interpolation='none')        
        ims.append([im])

    ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True)
    ani.save('animation.gif', writer='imagemagick', fps=12)
