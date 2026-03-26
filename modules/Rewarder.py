from OmniNaviPy.modules import Component
import numpy as np
import math

# cacluates reward/penalty based on given state variables
class Rewarder(Component.Component):
    
    # alpha is the constant weight assigned to each reward
    def __init__(self, alpha):
        self.alpha = alpha
        
    # how do we calculate reward?
    def calculate(self, environment):
        raise NotImplementedError

# penalizes each step taken
class Step(Rewarder):
    
    def step(self, environment):
        reward = self.calculate()
        return reward
        
    def calculate(self):
        reward = self.alpha
        return reward

# penalizes taking specific actions
class Actions(Rewarder):

    def __init__(self, alpha, action_values):
        super().__init__(alpha)
        self.action_values = action_values
    
    def step(self, environment):
        action_value = environment.get_action_value()
        reward = self.calculate(action_value)
        return reward
        
    def calculate(self, action_value):
        reward = 0
        if action_value in self.action_values:
            reward = self.alpha
        return reward

# updates state dictionary with terminate value if the maximum number of steps is reached
# steps_multiplier is used to determine max steps based on number of expected steps for current episode
class MaxSteps(Rewarder):
    
    def __init__(self, alpha, steps_multiplier):
        super().__init__(alpha)
        self.steps_multiplier = steps_multiplier
        
    def start(self, environment):
        path_steps = environment.get_path_steps()
        self.max_steps = math.ceil(self.steps_multiplier * path_steps)

    def step(self, environment):
        n_steps = environment.get_steps()
        reward = self.calculate(n_steps)
        return reward
        
    def calculate(self, n_steps):
        reward = 0
        if n_steps >= self.max_steps:
            reward = self.alpha
        return reward

# parent class that requires knowing the goal distance to calculate
class UsesGoalDistance(Rewarder):
    
    def step(self, environment):
        goal_distance = environment.get_goal_distance()
        reward = self.calculate(goal_distance)
        return reward
        
# penalizes moves that place the agent further from goal
class GoalDistance(UsesGoalDistance):
    
    def calculate(self, goal_distance):
        reward = self.alpha*goal_distance
        return reward

# updates state dictionary with terminate value if reached goal within threshold
class Goal(UsesGoalDistance):
    
    def __init__(self, alpha, goal_tolerance):
        super().__init__(alpha)
        self.goal_tolerance = goal_tolerance
        
    def calculate(self, goal_distance):
        reward = 0
        if goal_distance <= self.goal_tolerance:
            reward = self.alpha
        return reward