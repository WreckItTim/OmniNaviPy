from OmniNaviPy.modules import Environment
from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Agent
import numpy as np

# abstract template for sensors -- need methods filled by agent that defines how to sense the environment (acquire observations, etc)
# define a sense() function that returns a numpy array
# define a get_null() function that defines how to return a missing observation
class Sensor(Component.Component):
    
    def __init__(self, agent: Agent.Agent, dtype, DataTransformation: callable):
        self.agent = agent
        self.dtype = dtype
        self.DataTransformation = DataTransformation

    def sense(self, episode:Episode.Episode=None):
        raise NotImplementedError

    def get_shape(self):
        raise NotImplementedError

    # called at the end of every sense(), apply data transformation and convert to correct dtype
    def transform(self, observation):

        # apply data transformation if applicable
        if self.DataTransformation is not None:
            observation = self.DataTransformation.transform(observation)

        # convert to correct dtype
        observation = observation.astype(self.dtype)

        return observation

    def get_null(self):
        shape = self.get_shape()
        return np.zeros(shape, dtype=self.dtype)

# used to capture an RGB-D image from the agent
class Camera(Sensor):
    
    # camera_name is the name of the camera to capture from, which should correspond to a camera defined in the Agent's class
    def __init__(self, agent: Agent.Agent, camera_name: str, DataTransformation: callable=None, dtype: np.dtype=np.uint8):
        super().__init__(agent, dtype, DataTransformation)
        self.camera_name = camera_name

    def get_shape(self):
        sensor_meta = self.agent.get_sensor_meta(self.camera_name)
        sensor_shape = sensor_meta['shape']
        return sensor_shape

    def sense(self, episode=None):

        # get image from agent's camera
        image = self.agent.get_image(self.camera_name)

        # if image is None, return null observation, which is an array of zeros with the same shape as the sensor's observation space
        if image is None:
            image = self.get_null()

        return self.transform(image)

# gets relative x,y(,z) position of goal from agent's current position and orientation
# note that goal can either be an intermediate waypoint or the final target point
class RelativeGoal(Sensor):
    
    # self_normalize is a boolean that determines whether to self normalize the relative goal observation, which can help with training since the components of the relative goal (r and theta) are on different scales
    def __init__(self, agent: Agent.Agent, DataTransformation: callable=None, 
                    dtype: np.dtype=np.float32, self_normalize: bool=True, xyz: bool=True):
        super().__init__(agent, dtype, DataTransformation)
        self.self_normalize = self_normalize
        self.xyz = xyz

    def get_shape(self):
        if self.xyz:
            return (3,)
        return (2,)

    def sense(self, episode=None):

        # get displacement vector from agent to goal
        goal_point = episode.waypoint if episode.waypoint is not None else episode.target_point
        agent_point = self.agent.get_point()
        goal_displacement = goal_point.displacement(agent_point)

        # get polar coordinates from displacement
        r = np.linalg.norm(goal_displacement)
        theta = np.arctan2(goal_displacement[1], goal_displacement[0])
        if self.xyz:
            delta_z = goal_displacement[2]

        # self normalize since r and theta are on different scales
        if self.self_normalize:
            r = np.interp(r, (0, 255), (0.1, 1))
            theta = np.interp(theta, (-np.pi, np.pi), (0.1, 1))
            if self.xyz:
                delta_z = np.interp(delta_z, (-127, 127), (0.1, 1))
        
        # return relative goal as numpy array
        if self.xyz:
            relative_goal = np.array([r, theta, delta_z])
        else:
            relative_goal = np.array([r, theta])

        return self.transform(np.array(relative_goal))

# gets distance to nearest boundary in the direction the agent is moving
class DistanceBounds(Sensor):
    
    def __init__(self, agent: Agent.Agent, DataTransformation: callable=None,
                    dtype: np.dtype=np.float32):
        super().__init__(agent, dtype, DataTransformation)

    def get_shape(self):
        return (1,)

    def sense(self, episode=None):

        # get bounds of environment wrt current point of agent
        x_min, x_max, y_min, y_max, z_min, z_max = self.agent.get_bounds()
        
        # get direction agent is moving in
        # absolute facing was broken in version drl_beta (it was wrt yaw as opposed to movement)
        #moving_absolute = self.agent.get_moving_absolute()
        #print(x_min, x_max, y_min, y_max, z_min, z_max, f'moving_absolute: {moving_absolute}')

        # get position of agent
        point = self.agent.get_point()
        x, y, z, yaw = point.x, point.y, point.z, point.yaw

        # calculate distance to nearest boundary in the direction the agent is moving
        # if moving_absolute == 'Forward':
        #     bounds_distance = abs(y - (y_max-1))
        # elif moving_absolute == 'Backward':
        #     bounds_distance = abs(y - y_min)
        # elif moving_absolute == 'Right':
        #     bounds_distance = abs(x - (x_max-1))
        # elif moving_absolute == 'Left':
        #     bounds_distance = abs(x - x_min)
        # elif moving_absolute == 'up':
        #     bounds_distance = abs(z - (z_max-1))
        # elif moving_absolute == 'down':
        #     bounds_distance = abs(z - z_min)

        if yaw < 45 or yaw >= 315:
            bounds_distance = abs(x - (x_max-1))
        elif yaw >= 45 and yaw < 135:
            bounds_distance = abs(y - (y_max-1))
        elif yaw  >= 135 and yaw < 225:
            bounds_distance = abs(x - x_min)
        elif yaw  >= 225 and yaw < 315:
            bounds_distance = abs(y - y_min)
        
        # return distance to bounds as numpy array
        bounds_distance = np.array([bounds_distance])

        return self.transform(np.array(bounds_distance))

# gets current yaw of agent
class CurrentYaw(Sensor):
    
    def __init__(self, agent: Agent.Agent, DataTransformation: callable=None, dtype: np.dtype=np.float32):
        super().__init__(agent, dtype, DataTransformation)

    def get_shape(self):
        return (1,)

    def sense(self, episode=None):

        # get yaw from agent's current point
        yaw = self.agent.get_yaw()

        return self.transform(np.array([yaw]))