from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Action


# abstract templates -- need filled by agent that defines
    # how to act in the environment (move, rotate, etc)
    # how to sense the environment (acquire observations, etc)

# **** ABSTRACT AGENT CLASS ****
# Parent class to handle agent's actions and sensors -- an agent is a collection of actions and sensors that can interact with the environment
class Agent(Component.Component):
    def __init__(self):
        self.current_action = None

    def start(self, episode):
        self.current_action = None

    # ******** ABSTRACT ACTION METHODS ********

    # how to increment time forward during act() 
    def step(self):
        raise NotImplementedError

    # move forward wrt yaw (ignore pitch and roll) -- returns destination_point
    def move_forward(self, magnitude):
        raise NotImplementedError
    
    # strafe right wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_right(self, magnitude):
        raise NotImplementedError

    # strafe left wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_left(self, magnitude):
        raise NotImplementedError

    # rotate clockwise about the z-axis in place
    def rotate_clockwise(self, magnitude):
        raise NotImplementedError
    
    # rotate counter-clockwise about the z-axis in place
    def rotate_counter(self, magnitude):
        raise NotImplementedError
    
    # stop all motion immediately
    def stop(self):
        raise NotImplementedError

    # check if has collided
    def check_collision(self):
        raise NotImplementedError

    # check if a collision is imminent by checking if any distance sensors are below a certain threshold
    def check_collision_avoidance(self):
        raise NotImplementedError

    # check if is moving by checking if linear velocity, angular velocity, linear acceleration, or angular acceleration are above a certain threshold
    def is_moving(self, linear_threshold = 0.01, angular_threshold = 0.01, accel_threshold = 0.01):
        raise NotImplementedError

    # teleport to a point in the environment (for spawning at specific locations)
    def teleport(self, point):
        raise NotImplementedError

    # takeoff, and enter hover mode (for drones)
    def takeoff(self):
        raise NotImplementedError


    # ******** ABSTRACT SENSOR METHODS ********

    # returns a DataStructure.Point object representing the agent's current location and orientation in the environment    
    def get_point(self):
        raise NotImplementedError

    # returns the yaw angle in Euler angles of the agent in the environment (in degrees, between 0 and 360) where 0 is along the positive-x axis, and increases clockwise
    def get_yaw(self):
        raise NotImplementedError

    # returns a DataStructure.Point object representing the agent's current location and orientation in the simulator    
    def get_point(self):
        raise NotImplementedError

    # get rotation about the z-axis (yaw) in euler angles (0 degrees facing right, 90 degrees facing forward, 180 degrees facing left, 270 degrees facing backward) or airsim radians (0 radians facing forward, pi/2 radians facing right, pi radians facing backward, -pi/2 radians facing left)
    def get_yaw(self, as_euler=True):
        raise NotImplementedError

    # get image from drone's camera, camera_name is the name of the camera to capture from, which has predefined settings
    def get_image(self, camera_name):
        raise NotImplementedError

    # get metadata about the sensor, such as the shape of the observation space
    def get_sensor_meta(self, camera_name):
        raise NotImplementedError


    # ******** BASE HELPER METHODS ********

    # returns true if the current agent's state fulfills the episode's given objective (i.e. reaching a target point, etc)
    def check_objective(self, episode):
        point = self.get_point()
        distance = point.distance(episode.target_point)
        return distance < self.target_threshold

    # instructs agent to calculate movements in discrete space (1 meter and 90 degree increments)
    def set_discrete_space(self, value: bool):
        self.discrete_space = value

    # instructs agent to maintaint a fixed altitude during movements
    def set_fixed_z(self, value: float):
        self.fixed_z = value

    # defines x,y,z coordinates that the agent can move in
    def set_bounds(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def get_bounds(self):
        return self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max

    # checks if agent is out of bounds, if bounds are set
    def check_outofbounds(self, x=None, y=None, z=None):
        if hasattr(self, 'x_min'):
            if x is None or y is None or z is None:
                point = self.get_point()
                x, y, z = point.x, point.y, point.z
            if x-self.bounds_tolerance < self.x_min or x+self.bounds_tolerance >= self.x_max or y-self.bounds_tolerance < self.y_min or y+self.bounds_tolerance >= self.y_max or z-self.bounds_tolerance < self.z_min or z+self.bounds_tolerance >= self.z_max:
                return True
        return False

    # get disrete direction the drone is moving relative to yaw as a string
    def get_moving_relative(self):
        if isinstance(self.current_action, Action.Forward):
            return 'Forward'
        elif isinstance(self.current_action, Action.StrafeRight):
            return 'Right'
        elif isinstance(self.current_action, Action.StrafeLeft):
            return 'Left'
        elif isinstance(self.current_action, Action.RotateClockwise):
            return 'Clockwise'
        elif isinstance(self.current_action, Action.RotateCounter):
            return 'Counter'
        else:
            return 'Unknown' 

    # get disrete direction the drone is moving absolute to the environment as a string
    def get_moving_absolute(self):

        # set absolute direction moving in
        yaw = self.get_yaw()
        if isinstance(self.current_action, (Action.Forward, Action.RotateClockwise, Action.RotateCounter)):
            if yaw  < 45 or yaw >= 315:
                moving_absolute = 'Right'
            elif yaw >= 45 and yaw < 135:
                moving_absolute = 'Forward'
            elif yaw  >= 135 and yaw < 225:
                moving_absolute = 'Left'
            elif yaw  >= 225 and yaw < 315:
                moving_absolute = 'Backward'
        elif isinstance(self.current_action, (Action.StrafeRight)):
            if yaw  < 45 or yaw >= 315:
                moving_absolute = 'Backward'
            elif yaw >= 45 and yaw < 135:
                moving_absolute = 'Right'
            elif yaw  >= 135 and yaw < 225:
                moving_absolute = 'Forward'
            elif yaw  >= 225 and yaw < 315:
                moving_absolute = 'Left'
        elif isinstance(self.current_action, (Action.StrafeLeft)):
            if yaw  < 45 or yaw >= 315:
                moving_absolute = 'Forward'
            elif yaw >= 45 and yaw < 135:
                moving_absolute = 'Left'
            elif yaw  >= 135 and yaw < 225:
                moving_absolute = 'Backward'
            elif yaw  >= 225 and yaw < 315:
                moving_absolute = 'Right'
        else: # assume forward movement
            if yaw < 45 or yaw >= 315:
                moving_absolute = 'Right'
            elif yaw >= 45 and yaw < 135:
                moving_absolute = 'Forward'
            elif yaw  >= 135 and yaw < 225:
                moving_absolute = 'Left'
            elif yaw  >= 225 and yaw < 315:
                moving_absolute = 'Backward'

        return moving_absolute