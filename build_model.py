import random
import math

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import PReLU

from domain import *

class BuildModel:
    def __init__(self, dshape):
        """
        dshape: domain shape
        num_scenario: different configuration of wall/gold/storage
        num_data: for each scenario, play num_data times.  
        """
        self.dshape = dshape
        self.nrows, self.ncols = self.dshape
        self.model_size = self.nrows * self.ncols
        self.mydomain = Domain((self.nrows,self.ncols))
        self.learning_rate = 0.001
        self.gamma = 0.9

    def _build_domain(self, p_wall, p_gold):
        
        mywall = []
        mystorage = []
        mygold=[]

        n_walls = math.floor(p_wall * (self.nrows * self.ncols)) 
        n_golds = math.floor(p_gold * (self.nrows * self.ncols)) 
        n_storages = math.floor(1.2 * n_golds)

        cubes = []

        for i in range(self.nrows):
            for j in range(self.ncols):
                cubes.append([i,j]) 

        random.shuffle(cubes)
        
        walls = []
        for _ in range(n_walls):
            walls.append(cubes.pop())

        self.mydomain.add_wall(walls)

        storages = []
        for _ in range(n_storages):
            storages.append(cubes.pop())

        self.mydomain.add_storage(storages)

        golds = []
        for _ in range(n_golds):
            golds.append(cubes.pop())

        self.mydomain.add_gold(golds)

        agent_loc = cubes.pop()
        self.mydomain.add_agent([agent_loc])

    def plot_it(self):
        self.mydomain.plot_domain(True)


    def _init_play(self, num_data, p_wall, p_gold):
        self._build_domain(p_wall,p_gold)

        init_data = []
        for i in range(num_data):
            a = random.randint(0,3)
            current_state = self.mydomain.get_state()
            reward, done = self.mydomain.action(self.mydomain.actions[a])
            next_state = self.mydomain.get_state()
            episode = [current_state, reward, a, next_state]
            tr_data = self._raw_data_to_training(episode)
            init_data.append(tr_data)


        return init_data
   

    def _build_model(self):

        # create model
        prediction_model = Sequential()

        # add model layers
        prediction_model.add(Dense(self.model_size, input_shape=(self.model_size,)))
        prediction_model.add(PReLU())
        prediction_model.add(Dense(self.model_size))
        prediction_model.add(PReLU())
        prediction_model.add(Dense(len(self.mydomain.actions)))
        prediction_model.compile(optimizer='adam', loss='mse')
        
        self.prediction_model = prediction_model

    def _train_it(self):
        data = self._init_play(1000,0.03,0.05)
        input_1 = data[0][0]
        label_1 = data[1][1]
        for i in range(1,len(data)):
            input_1 = np.vstack([input_1,data[i][0]])
            label_1 = np.vstack([label_1,data[i][1]])

        self.prediction_model.fit(input_1,label_1, epochs=10, batch_size=100, verbose=0)

        


    def _raw_data_to_training(self, mydata):
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
        target_cur = self.prediction_model.predict(current_state)[0]
        target_next = self.prediction_model.predict(next_state)[0]
        
        # Second modify the Q value of the action you took. 
        target_cur[action] = (1 - self.learning_rate) * target_cur[action] + self.learning_rate*(instant_reward + self.gamma * (max(target_next)))

        return [current_state[0],target_cur]       


    def get_initial_predition_model(self):
        self._build_domain(p_wall = 0.2, p_gold=0.1)
        self._build_model()
        self._train_it()

        return self.prediction_model


if __name__ == "__main__":
    bm = BuildModel((10,10))
    mymodel = bm.get_initial_predition_model()
    print(mymodel.summary())