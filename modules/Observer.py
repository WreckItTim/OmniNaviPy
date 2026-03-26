from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import Sensor
import numpy as np

# parent class that handles how to process a numpy array of observations
# process() inputs numpy array of observations and outputs processed observations
# assumes each observation is of size (n,dim_1,...,dim_2)
class Observer(Component.Component):

    # constructor inputs list of sensor objects to get various needed info from
    # n_history keeps track of previous many frames of observations
    def __init__(self, sensors:list[Sensor.Sensor], n_history:int=1, data_type=np.float32):
        self.sensors = sensors
        self.n_history = n_history
        self.data_type = data_type
        self.calculate_output_shape()

    # initialize history to zeros (zero should be reserved for missing observations during normalization)
    def start_history(self):
        self.history = np.zeros(self.history_shape, dtype=self.data_type)

    def start(self, episode:Episode.Episode):
        for sensor in self.sensors:
            sensor.start(episode)
        self.start_history()

    def end(self, episode:Episode.Episode):
        for sensor in self.sensors:
            sensor.end(episode)

    # goes through sensor objects to aggergate returned sizes
    def calculate_output_shape(self):
        sensor_shapes = []
        for sensor in self.sensors:
            sensor_shape = list(sensor.get_shape())
            sensor_shapes.append(sensor_shape)
        self.n_elements = sum([sensor_shape[0] for sensor_shape in sensor_shapes])
        extra_dims = sensor_shapes[0][1:]
        self.frame_shape = tuple([self.n_elements] + extra_dims)
        self.history_shape = tuple([self.n_history*self.n_elements] + extra_dims)
        self.is_vector = False
        self.o_type = 'array'
        if len(self.frame_shape) == 1:
            self.is_vector = True
            self.o_type = 'vector'

    # acquires sensor data from each sensor and pushes it onto the history array that is a FIFO queue
    def observe(self, episode:Episode.Episode=None):
        # rotate sliding history
        for i in range(self.n_history-1, 0, -1):
            old_frame = self.history[(i-1)*self.n_elements:i*self.n_elements]
            self.history[i*self.n_elements:(i+1)*self.n_elements] = old_frame
        # get fresh observations
        observations = []
        for idx, sensor in enumerate(self.sensors):
            observation = sensor.sense(episode)
            observations.append(observation)
        # aggregrate fresh observations and integrate with history array
        if self.is_vector:
            new_frame = np.hstack(observations)
        else:
            new_frame = np.vstack(observations)
        self.history[:self.n_elements] = new_frame
        return self.history

# outputs a dictionary of process observations
class DictObserver(Observer):

    # observer_dict is {name:observer} {str:Observer}
    # observer_dict is {name:idxs} {str:list<int>} 
        # where idxs correspond to observations numpy array in process()
    def __init__(self, observer_dict):
        self.observer_dict = observer_dict
        self.output_shape_dict = self.calculate_output_shape()
        self.o_type = 'dict'

    # goes through sensor objects to aggergate return size
    def calculate_output_shape(self):
        output_shape_dict = {}
        for name in self.observer_dict:
            observer = self.observer_dict[name]
            output_shape_dict[name] = observer.history_shape
        return output_shape_dict

    def start(self, episode:Episode.Episode):
        for name in self.observer_dict:
            observer = self.observer_dict[name]
            observer.start(episode)

    def end(self, episode:Episode.Episode):
        for name in self.observer_dict:
            observer = self.observer_dict[name]
            observer.end(episode)

    # observe from each observer and return as dictionary of observations
    def observe(self, episode:Episode.Episode=None):
        observations_dict = {}
        for name in self.observer_dict:
            observer = self.observer_dict[name]
            observations = observer.observe(episode)
            observations_dict[name] = observations
        return observations_dict

