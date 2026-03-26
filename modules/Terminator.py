from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Agent
import numpy as np
import math

DEFAULT_PRINT_TERMINATION = False

# used to determine if we terminate an episode
class Terminator(Component.Component):

    def __init__(self, print_termination: bool=DEFAULT_PRINT_TERMINATION):
        self.print_termination = print_termination
    
    # how do we check for termination?
    def check(self):
        raise NotImplementedError

# terminate if reached goal within threshold
class Goal(Terminator):
    
    def __init__(self, agent: Agent.Agent, goal_tolerance: float, print_termination: bool=DEFAULT_PRINT_TERMINATION):
        super().__init__(print_termination)
        self.agent = agent
        self.goal_tolerance = goal_tolerance
        
    # check distance between agent and goal
    def check(self, episode: Episode.Episode):
        goal_distance = episode.target_point.distance(self.agent.get_point())
        terminate = goal_distance <= self.goal_tolerance
        if terminate:
            if self.print_termination:
                print('TERMINATED EPISODE, REACHED TARGET POSITION')
        return terminate, 'goal_reached'

# terminate if the maximum number of steps is reached
# steps_multiplier is used to determine max steps based on expected steps for current episode
class MaxSteps(Terminator):
    
    def __init__(self, steps_multiplier, print_termination: bool=DEFAULT_PRINT_TERMINATION):
        super().__init__(print_termination)
        self.steps_multiplier = steps_multiplier
        
    def start(self, episode: Episode.Episode):
        self.max_steps = math.ceil(self.steps_multiplier * episode.ground_truth_trajectory.n_steps())
        
    def check(self, episode: Episode.Episode):
        n_steps = episode.n_steps()
        terminate = n_steps >= self.max_steps
        if terminate:
            if self.print_termination:
                print('TERMINATED EPISODE, VIOLATED THE TIME CONSTRAINT')
        return terminate, 'max_steps_exceeded'