
import random 

from domain import *


# function to create a random domain

def create_random_domain(domain_shape,num_wall, num_storage, num_gold):

    my_domain = Domain(domain_shape)
    
 
    available_loc = []

    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            available_loc.append([i,j])

    # shuffle the list of available locations
    random.shuffle(available_loc)


    walls = []
    storages = []
    golds=[]
    
    
    # assign walls
    for _ in range(num_wall):
        tmp = available_loc.pop()
        my_domain.add_wall([tmp])
    

    # assign storages
    for _ in range(num_storage):
        tmp = available_loc.pop()
        my_domain.add_storage([tmp])
    
    # assign golds
    for _ in range(num_gold):
        tmp = available_loc.pop()
        my_domain.add_gold([tmp])
    

    tmp = available_loc.pop()
    my_domain.add_agent([tmp])

    return my_domain



if  __name__ == '__main__':
    
    md = create_random_domain((10,10),5,4,3)
    md.plot_domain(True)