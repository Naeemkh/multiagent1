
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
        "storage": [3,[255,224,224]]
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
            (new_row, new_col, pickup, dropoff) = self.agent.take_action(action_item)

            if pickup:
                # look see if there is a gold in the box and if there is pick it up, reward, and update accordingly.
                if self.domain_mat[self.agent.row][self.agent.col].ctype == "gold" and not self.agent.has_box:
                    self.agent.has_box = True
                    self.domain_mat[self.agent.row][self.agent.col].update("fspace")
                    
            elif dropoff:
                # look see if it is a dropoff location
                # Agent can drop the box at any place, however, if the place is storage, should get a good reward.
                if self.domain_mat[self.agent.row][self.agent.col].ctype != "gold" and self.agent.has_box:
                    self.agent.has_box = False
                    self.domain_mat[self.agent.row][self.agent.col].update("gold")

            elif  (new_row < 0 or new_row > self.nrows-1) or (new_col < 0 or new_col > self.ncols-1):
                # the agent wants to go out. Should be given penalty for going out of the board.
                pass
                
            elif self.domain_mat[new_row][new_col].ctype == "wall":
                # the agent wants to go into the wall should be given penalty
                pass

            elif self.domain_mat[new_row][new_col].ctype == "gold":
                # the agent wants to go into a place with gold. 
                # there are two possiblities:
                # 1) if it carries gold, give penalty, because the gold should go to the storage.
                # 2) if it does not carry a gold, give some reward.
                
                self.agent.row = new_row
                self.agent.col = new_col

            # elif self.domain_mat[new_row][new_col].ctype == "storage":
            #     # the agent wants to go into a place which is designated as a storage.
            #     # there are two possiblities:
            #     # 1) if it carries gold, give some reward.
            #     # 2) if it does not carry a gold, give penalty.
            #     pass

            else:
                # the agent goes some where normal, it will get small negative reward because of wandering. 
                self.agent.row = new_row
                self.agent.col = new_col

            
    def update_status(self,new_stat):
        pass

    def provide_stat(self):
        pass

    def is_action_valid(self):
        pass

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
        

    def current_state(self):
        pass

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

    for i in range(5,12):
        mywall.append([i,18])

    for j in range(5,12):
        mywall.append([10,j])
     
    storage = [[19,6],[19,7],[19,8],[19,9],[19,10],[19,11],[19,12],[19,13]]
    mydomain.add_wall(mywall)
    
    mygold=[]
    for i in range(1,2):
        for j in range(1,20): 
            mygold.append([i,j])

    for i in range(4,5):
        for j in range(1,20): 
            mygold.append([i,j])

    for i in range(20,21):
        for j in range(1,20): 
            mygold.append([i,j])

    for i in range(15,16):
        for j in range(1,20): 
            mygold.append([i,j])

    mydomain.add_gold(mygold)
    mydomain.add_storage(storage)
    mydomain.add_agent([[4,11]])

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

    for i in range(5000):
        a = random.randint(0,5)
        mydomain.action(actions[a])
        im = plt.imshow(mydomain.plot_domain(False), interpolation='none')        
        ims.append([im])

    ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True)
    ani.save('animation.gif', writer='imagemagick', fps=12)
