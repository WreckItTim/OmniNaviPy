from OmniNaviPy.modules import Environment
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Utils
import os

# Abstract calss that runs a loop for a given environment, and handles checkpointing
class Run:
    def __init__(self, environment, ckpt_freq=None, ckpt_dir=None):
        self.environment = environment
        self.ckpt_freq = ckpt_freq
        self.ckpt_dir = ckpt_dir

    # checkpoint handling
    def checkpoint_in(self, ckpt_dir):
        # read ckpt params for all components
        self.environment.spawner.checkpoint_in(ckpt_dir)
        self.environment.actor.checkpoint_in(ckpt_dir)
        self.environment.observer.checkpoint_in(ckpt_dir)
        for terminator in self.environment.terminators:
            terminator.checkpoint_in(ckpt_dir)
        for other in self.environment.others:
            other.checkpoint_in(ckpt_dir)
    def checkpoint_out(self, ckpt_dir):
        # write ckpt params from all components
        self.environment.spawner.checkpoint_out(ckpt_dir)
        self.environment.actor.checkpoint_out(ckpt_dir)
        self.environment.observer.checkpoint_out(ckpt_dir)
        for terminator in self.environment.terminators:
            terminator.checkpoint_out(ckpt_dir)
        for other in self.environment.others:
            other.checkpoint_out(ckpt_dir)

    def run(self):
        raise NotImplementedError('Run is an abstract class, please use a subclass with a defined run() method')

# runs a loop for a given environment, and handles checkpointing
class Evaluate(Run):

    def run(self, write_path=None, overwrite=False, start_path_idx=0, end_path_idx=None, 
                save_observations=False, view_live_plt=True, state_map_path=None):

        # reset spawner -- sets any vars as needed to check if there are more paths to evaluate
        self.environment.spawner.reset()

        # keep playing a new episode while spawner has more paths to evaluate on
        if write_path is not None and os.path.exists(write_path) and not overwrite:
            episodes = Utils.pickle_read(write_path)
        else:
            episodes = {}

        # set begining and ending spawner index
        path_idx = start_path_idx + len(episodes)
        if end_path_idx is None:
            end_path_idx = self.environment.spawner.n_paths()

        # iterate through each episode
        if path_idx < end_path_idx:
            self.environment.spawner.skip_to(path_idx)
        while(path_idx < end_path_idx):
            episode = self.environment.play_episode(save_observations, view_live_plt, state_map_path)
            episodes[path_idx] = episode

            # checkpoint current evaluations?
            if self.ckpt_freq is not None and path_idx % self.ckpt_freq == 0:
                self.checkpoint_out(self.ckpt_dir)
                Utils.pickle_write(write_path, episodes)
                # navigation accuracy -- percent that reached goal within a certain threshold distance
                n_successes = 0
                for path_idx in episodes:
                    episode = episodes[path_idx]
                    termination = episode.termination
                    if termination == 'goal_reached':
                        n_successes += 1
                accuracy = 100 * n_successes / len(episodes)
                print(f'completed {len(episodes)} episodes with accuracy {accuracy:.2f}%')

            path_idx += 1

        # write final episodes and other checkpoints to file
        if self.ckpt_freq is not None:
            self.checkpoint_out(self.ckpt_dir)
        Utils.pickle_write(write_path, episodes)

        return episodes