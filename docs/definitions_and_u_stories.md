# Definitions

- Project: Multi-Agent Reinforcement Learning

## Classes

### ** Agent **
- The Agent class is used to generate an agent instance.
- It keeps the agent's location and whether it has the box or not.
- At this point, we only create a single agent. However, in the future, the class will be necessary for the multiagent system.

### ** Cube **
- Each grid point on the study region is an instance of a Cube class. 
- The Cube class handles the different tasks with the cube instance. 
- For example, the type of cube, color, code and so on. 

### ** Domain **
- This class includes all information regarding the domain. 
- Each domain can have one or more agents. 
- Each domain can have one or more storages.
- Each domain includes the location of gold nuggets, walls, and so on.
- An agent can try one action out of 6 actions:
    - Up
    - Down
    - Left
    - Right
- However, the domain instance decides if the action is a valid action  and reward or punish the agent based on the situation.
