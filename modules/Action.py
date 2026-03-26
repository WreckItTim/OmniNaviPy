from OmniNaviPy.modules import Component

PRINT_ACTION_DEFAULT = True

# abstract template when requires functions act() and an agent to use to act on the environment, and step() function to update the environment based on the action taken
class Action(Component.Component):

    def __init__(self, agent, print_action:bool=PRINT_ACTION_DEFAULT):
        self.agent = agent
        self.print_action = print_action

    # uses agent to interact with environment
    def act(self):
        self.start_point = self.agent.get_point() # get starting point to compare against for determining when action is done
        self.agent.current_action = self # set current action for agent

    def __repr__(self):
        return str(self)

class TranlationalAction(Action):
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, print_action)
        self.magnitude = magnitude
    # check if agent has moved by the given magnitude within a given threshold
    def is_done(self):
        point = self.agent.get_point()
        distance = point.distance(self.destination_point)
        #print(point, 'to', self.destination_point, 'distance:', distance)
        reached_point = distance <= self.agent.translational_threshold
        moving = self.agent.is_moving()
        return reached_point or not moving

class RotationalAction(Action):
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, print_action)
        self.magnitude = magnitude
    # check if agent has rotated by the given magnitude within a given threshold
    def is_done(self):
        #delta_angle = abs(self.agent.get_point().yaw - self.start_point.yaw)
        #return delta_angle < self.agent.rotational_threshold
        return True # handle this syncrhonously for now as rotations self adjust with PID

# moves agent in the forward facing direction    
class Forward(TranlationalAction):
    # magnitude is in meters
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, magnitude, print_action)
    def act(self):
        super().act()
        if self.print_action:
            print('move forward', self.magnitude, 'meters')
        self.destination_point = self.agent.move_forward(self.magnitude)
    def __str__(self):
        return f'Forward{self.magnitude}'

# moves agent in the right direction relative to facing direction
class StrafeRight(TranlationalAction):
    # magnitude is in meters
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, magnitude, print_action)
    def act(self):
        super().act()
        if self.print_action:
            print('strafe right', self.magnitude, 'meters')
        self.destination_point = self.agent.strafe_right(self.magnitude)
    def __str__(self):
        return f'StrafeRight{self.magnitude}'

# moves agent in the left direction relative to facing direction
class StrafeLeft(TranlationalAction):
    # magnitude is in meters
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, magnitude, print_action)
    def act(self):
        super().act()
        if self.print_action:
            print('strafe left', self.magnitude, 'meters')
        self.destination_point = self.agent.strafe_left(self.magnitude)
    def __str__(self):
        return f'StrafeLeft{self.magnitude}'
    
# rotates agent clockwise in place (around z axis)
class RotateClockwise(RotationalAction):
    # magnitude is in degrees
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, magnitude, print_action)
    def act(self):
        super().act()
        if self.print_action:
            print('rotate clockwise', self.magnitude, 'degrees')
        self.agent.rotate_clockwise(self.magnitude)
    def __str__(self):
        return f'RotateClockwise{self.magnitude}'
    
# rotates agent counter-clockwise in place (around z axis)
class RotateCounter(RotationalAction):
    # magnitude is in degrees
    def __init__(self, agent, magnitude: int, print_action:bool=PRINT_ACTION_DEFAULT):
        super().__init__(agent, magnitude, print_action)
    def act(self):
        super().act()
        if self.print_action:
            print('rotate counter-clockwise', self.magnitude, 'degrees')
        self.agent.rotate_counter(self.magnitude)
    def __str__(self):
        return f'RotateCounter{self.magnitude}'