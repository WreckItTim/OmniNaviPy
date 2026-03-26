from OmniNaviPy.modules import DataStructure
from OmniNaviPy.modules import Trajectory
from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Agent
from OmniNaviPy.modules import Utils
from pathlib import Path
import random

DEFAULT_PRINT_SPAWN = False

# creates a new episode object on spawn() and teleports agent to given start point
class Spawner(Component.Component):

    def __init__(self, agent:Agent.Agent, trajectories:list[Trajectory.Trajectory], print_spawn:bool=DEFAULT_PRINT_SPAWN,
                    random_spawn:bool=False):
        self.agent = agent
        self.trajectories = trajectories
        self.print_spawn = print_spawn
        self.random_spawn = random_spawn
        self.reset()

    def checkpoint_in(self, ckpt_dir):
        self.trajectory_idx = Utils.pickle_read(Path(ckpt_dir, 'Spawner__trajectory_idx.p'))
    
    def checkpoint_out(self, ckpt_dir):
        Utils.pickle_write(Path(ckpt_dir, 'Spawner__trajectory_idx.p'), self.trajectory_idx)

    def reset(self):
        self.trajectory_idx = 0
        
    # increment to next trajectory
    def end(self, episode:Episode.Episode):
        self.trajectory_idx += 1
        if self.trajectory_idx >= len(self.trajectories):
            self.trajectory_idx = 0 # loop back around to beginning of trajectories if we reach the end
    
    def spawn(self, save_observations:bool=False):

        # get trajectory at next index
        if self.random_spawn:
            trajectory = random.choice(self.trajectories)
        else:
            trajectory = self.trajectories[self.trajectory_idx]

        # teleport agent to start point of trajectory
        start_point = trajectory.points[0]
        self.agent.teleport(start_point)

        if self.print_spawn:
            print(f"Spawned at {start_point}")

        # make episode object
        target_point = trajectory.points[-1]
        episode = Episode.Episode(start_point=start_point, target_point=target_point, 
                                    ground_truth_trajectory=trajectory, save_observations=save_observations)

        return episode

    # determine number of paths avaialble
    def n_paths(self):
        return len(self.trajectories)

    # skip to given trajectory at index
    def skip_to(self, trajectory_idx):
        if trajectory_idx >= 0 and trajectory_idx < len(self.trajectories):
            self.trajectory_idx = trajectory_idx
        else:
            raise IndexError(f"Trajectory index {trajectory_idx} is out of range for available trajectories of length {len(self.trajectories)}")


# samples from a set of ground truth trajectories, and incrementally levels up difficulty to sample from
class Curriculum(Spawner):

    # trajectories_dictionary is {difficulty: list of trajectories}
        # NOTE: it is assumed that the key order is sorted from lowest to highest difficulty
    # lower_difficulty_proba is probability to sample from a lower difficulty than the current one (set to zero to disable this)
    def __init__(self, agent:Agent.Agent, trajectories_dictionary:dict, level_up_freq:int, 
                    burn_in:int=0, lower_difficulty_proba:float=0.3, print_spawn:bool=DEFAULT_PRINT_SPAWN):
        self.agent = agent
        self.trajectories_dictionary = trajectories_dictionary
        self.lower_difficulty_proba = lower_difficulty_proba
        self.difficulties = list(self.trajectories_dictionary.keys())
        self.difficulty_index = 0
        self.level_up_freq = level_up_freq
        self.burn_in = burn_in
        self.print_spawn = print_spawn
        self.n_episodes = 0

    def checkpoint_in(self, ckpt_dir):
        self.difficulty_index = Utils.pickle_read(Path(ckpt_dir, 'Spawner__difficulty_index.p'))
        self.n_episodes = Utils.pickle_read(Path(ckpt_dir, 'Spawner__n_episodes.p'))
    
    def checkpoint_out(self, ckpt_dir):
        Utils.pickle_write(Path(ckpt_dir, 'Spawner__difficulty_index.p'), self.difficulty_index)
        Utils.pickle_write(Path(ckpt_dir, 'Spawner__n_episodes.p'), self.n_episodes)

    # go to next difficulty, unless nomore difficulties left, then signal finished
    def level_up(self):
        self.difficulty_index += 1
        if self.difficulty_index < len(self.difficulties):
            self.difficulty = self.difficulties[self.difficulty_index]
        else:
            self.difficulty = 'finished'

    # get list of difficulties that we can sample trajectories from
    def get_selectable_difficulties(self, max_difficulty:int):
        max_idx = self.difficulties.index(max_difficulty)
        difficulties = [difficulty for difficulty in self.difficulties[:max_idx]]
        return difficulties

    # determine if we level up to next difficulty at end of episode
    def end(self, episode:Episode.Episode):
        self.n_episodes += 1
        if self.n_episodes >= self.burn_in and self.n_episodes % self.level_up_freq == 0:
            self.level_up()
        
    # randomly fetch next path
    def spawn(self, save_observations:bool=False):
        difficulty = self.difficulty

        # if finished then sample from any difficulty
        if difficulty == 'finished':
            difficulty = random.choice(self.difficulties)

        # else check if we sample from lower difficulty or current one
        elif self.lower_difficulty_proba > 0 and random.random() < self.lower_difficulty_proba:
            selectable_difficulties = self.get_selectable_difficulties(difficulty)
            if len(selectable_difficulties) > 0:
                difficulty = random.choice(selectable_difficulties)

        # sample from trajectories at selected difficulty
        trajectories = self.trajectories_dictionary[difficulty]
        trajectory = random.choice(trajectories)

        # teleport agent to start point of trajectory
        start_point = trajectory.points[0]
        self.agent.teleport(start_point)

        if self.print_spawn:
            print(f"Spawned at {start_point}")

        # make episode object
        target_point = trajectory.points[-1]
        initial_state = {} # any other info?
        episode = Episode.Episode(start_point=start_point, initial_state=initial_state, target_point=target_point, 
                                    ground_truth_trajectory=trajectory, save_observations=save_observations)

        return episode

    # determine number of paths avaialble
    def n_paths(self):
        n_paths = 0
        for difficulty in self.difficulties:
            n_paths += len(self.trajectories_dictionary[difficulty])
        return n_paths