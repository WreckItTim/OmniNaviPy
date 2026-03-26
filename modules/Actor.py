from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Action
from OmniNaviPy.modules import Agent

PRINT_ENDING_DEFAULT = True

# translates action-value to actions from RLenvironment
class Actor(Component.Component):
    def __init__(self, actions: list[Action.Action], print_ending:bool=PRINT_ENDING_DEFAULT):
        self.actions = actions
        self.print_ending = print_ending

    def start(self, episode:Episode.Episode):
        for action in self.actions:
            action.start(episode)

    def end(self, episode:Episode.Episode):
        for action in self.actions:
            action.end(episode)

    # step through an action, checking for collisions and other termination conditions
    def step_through(self, episode:Episode.Episode, action:Action):
        agent = action.agent
        clear_collision = agent.check_collision()
        while(True):
            #if self.print_ending:
            #    print(f'Agent at point: {agent.get_point()}')
            if agent.check_collision():
                if self.print_ending:
                    print('Ending motion, due to collision')
                agent.stop()
                break
            # elif agent.check_objective(episode):
            #     if self.print_ending:
            #         print('Ending motion, because objective fulfilled')
            #     agent.stop()
            #     break
            elif action.is_done():
                if self.print_ending:
                    print('Ending motion, because action is done')
                break   
            elif agent.check_collision_avoidance():
                if self.print_ending:
                    print('Ending motion, because collision avoidance triggered')
                agent.stop()
                break
            elif agent.check_outofbounds():
                if self.print_ending:
                    print('Ending motion, because out of bounds')
                agent.stop()
                break
            agent.step() # tell agent to make a step, updating state

# chooses one action from those available
class Discrete(Actor):
    
    def act(self, action_value: int, episode:Episode.Episode):
        action = self.actions[int(action_value)]
        episode.add_action({'action_name': str(action), 'action_value': action_value})
        action.act() # run asynchronously or synchronously depending on the implementation of the action
        self.step_through(episode, action) # if asynchronous, then step through the action and check for termination conditions until action is done, otherwise this will just check if the action is done immediately after acting once