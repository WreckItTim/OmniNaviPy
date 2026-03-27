from OmniNaviPy.modules import Utils

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

PRINT_STEP = False

# define an environment which we can move an agent through
# considers episodic movement that progresses one step at a time until termination criteria is met
# one episode uses the class methods: start(), step(), ..., step(), end()
# this is heavily inspired by OpenAI Gymnaisum environment objects
    # see ReinforcementLearning class below and SB3Wrappers.py module to be fully compatible
# self.calculations={} contains key-value pairs of intermediate calculations
    # this can be used to avoid reduntant calculations for efficiency
class Episodic:
    # policy is the model that generates action_values based on observations
    # Component parameters have the parent-inherited methods: start(), step(), and end() which take environment as input
    # spawner, actions, observer, terminators, and others are components
        # spawner.start() sets initial environment x, y, z, direction, (target) values 
        # actor.step() updates intermediate environment x, y, z, direction values
        # observer.step() returns a processed observation captured at current state
        # terminate.step() returns a boolean determining if we should end the episode
        # other is used to trigger miscellaneous functionality on start/step/end as user defined
    def __init__(self, agent, policy, spawner, actor, observer, terminators, others=[]):
        self.agent = agent
        self.policy = policy
        self.spawner = spawner
        self.actor = actor
        self.observer = observer
        self.terminators = terminators
        self.others = others
        self.view_initialized = False
        
    # step through an entire episode with environment, generationg actions from model
    # model can be any object that has a predict() function as shown below
    # save_observations will save observations in self.current_episode.observations for each step (memory heavy)
    # if fig is not None that will visualize each episode to screen
    # state_map_path is an optional path to a visual map of the environment that can be displayed during play_episode() if provided
    def play_episode(self, save_observations=False, view_live_plt=False, state_map_path=None):

        # displays live observations and state map if using MLLM
        if view_live_plt and not self.view_initialized:
            self.initialize_view(state_map_path)

        # start episode
        episode = self.start(save_observations)

        # iterate through each step
        terminate = False
        while(not terminate):
        
            # take step based on action value
            terminate = self.step(episode)

            # initialize and log state variables
            if view_live_plt:
                self.update_view(episode, state_map_path)

        # end episode
        self.end(episode)
        
        return episode

    # used to visualize episode step-by-step
    def initialize_view(self, state_map_path=None):
        plt.ion()
        if state_map_path is None:
            self.fig, ax = plt.subplots(ncols=1,)
            depth_map_ax = ax
        else:
            self.fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 1.5]})
            depth_map_ax = axs[0]
            state_map_ax = axs[1]
            state_map = np.zeros((480, 680, 3), dtype=np.uint8)
            self.state_map_im = state_map_ax.imshow(state_map, aspect='auto')
            state_map_ax.axis('off')
            state_map_ax.set_title('Global State Map')
        depth_map = np.zeros((144, 256), dtype=np.uint8)
        self.depth_map_im = depth_map_ax.imshow(depth_map, vmin=0, vmax=255, cmap='grey', aspect='auto')
        depth_map_ax.axis('off')
        depth_map_ax.set_title('Local Depth Map')
        self.view_initialized = True
    def update_view(self, episode, state_map_path=None):
        depth_map = episode.get_depth_map()
        self.depth_map_im.set_data(depth_map)
        if state_map_path is not None and os.path.exists(state_map_path):
            state_map = mpimg.imread(state_map_path)
            self.state_map_im.set_data(state_map)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.tight_layout()
    
    #### EPISODIC CONTROL
    
    # start a new episode
    def start(self, save_observations=False):

        # query spawner to set initial state of agent and environment
        episode = self.spawner.spawn(save_observations)
        self.actor.start(episode)
        self.observer.start(episode)
        for terminator in self.terminators:
            terminator.start(episode)
        for other in self.others:
            other.start(episode)
        self.agent.start(episode)

        if PRINT_STEP:
            print(f'Starting episode at point: {episode.point} with target: {episode.target_point}')

        return episode

    # intermediate steps of episode
    def step(self, episode):

        # start a new step
        episode.new_step() 
        point = self.agent.get_point()
        episode.add_point(point)

        if PRINT_STEP:
            print(f'Starting step at point: {point}')

        # make observations
        observations = self.observer.observe(episode)
        episode.add_observations(observations)

        #if PRINT_STEP:
        #    print(f'Observations: {observations}')

        # predict action
        action_value = self.policy.predict(observations)

        # take action (actor will add the action to episode based on actor's defined behavior)
        self.actor.act(action_value, episode) # sets point at each action.step()

        if PRINT_STEP:
            action_str = episode.action['action_name'] 
            print(f'action_value: {action_value} -> action: {action_str}')

        # call step for any other components (which may or may not update episode, this object is intentially arbitrary for the user to define as they wish)
        for other in self.others:
            other.step(episode)
        
        # check if we should terminate episode
        terminate = False
        for terminator in self.terminators:
            terminate, reason = terminator.check(episode)
            if terminate:
                episode.add_termination(reason)
                break
        if PRINT_STEP:
            print(f'Terminate: {terminate}')
            if terminate:
                print(f'reason: {reason}')
            #pause = input('Press Enter to continue to next step...') # pause after each step for easier visualization

        return terminate

    # end of epsiode
    def end(self, episode):
        
        # end all components
        self.spawner.end(episode)
        self.actor.end(episode)
        self.observer.end(episode)
        for terminator in self.terminators:
            terminator.end(episode)
        for other in self.others:
            other.end(episode)
        self.agent.end(episode)



# # an Episodic environment enhanced for reinforcment learning (has additional attributes needed for training)
# class ReinforcementLearning(Episodic):
#     # rewarders are additional components to Episodic parent class
#         # rewarder.step() returns a reward/penalty float value
#     def __init__(self, spawner, actor, observer, terminators, rewarders, 
#                  others=[], ckpt_freq=None, ckpt_dir=None):
#         super().__init__(spawner, actor, observer, terminators, 
#                          others, ckpt_freq, ckpt_dir)
#         self.rewarders = rewarders

#     # checkpoint handling rewarders
#     def checkpoint_in(self, ckpt_dir):
#         super().checkpoint_in(ckpt_dir)
#         for rewarder in self.rewarders:
#             rewarder.checkpoint_in(ckpt_dir)
#     def checkpoint_out(self):
#         super().checkpoint_out()
#         for rewarder in self.rewarders:
#             rewarder.checkpoint_out(self.ckpt_dir)

#     # we need to start the rewarders
#     def start(self):
#         observations = super().start()
        
#         # start DRL components
#         for rewarder in self.rewarders:
#             rewarder.start(self.current_episode)
            
#         return observations
        
#     # we need to step through the rewarders, and return calculated reward
#     def step(self, action_value):
#         observations, terminate = super().step(action_value)

#         # reward function
#         total_reward = 0
#         reward_dict = {}
#         for rewarder in self.rewarders:
#             total_reward += rewarder.calculate(self.current_episode)
        
#         return observations, total_reward, terminate
        
#     # we need to end the rewarders and evaluators
#     def end(self):
#         # stop DRL components
#         for rewarder in self.rewarders:
#             rewarder.end(self)
#         # super end after reward end to trigger rewarder.end() before potential checkpoint
#         super().end()