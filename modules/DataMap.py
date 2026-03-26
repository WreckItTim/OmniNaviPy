from OmniNaviPy.modules import DataStructure
from OmniNaviPy.modules import Utils
from OmniNaviPy.modules import Agent
from pathlib import Path
import numpy as np
import math
import os

DATA_DIR = Utils.get_global('data_directory')

# get and write meta data from data dictionaries for given map and sensor
# this leads to quicker load times during execution of data fetches
def write_file_map(map_name, sensor_name):
    sensor_dir = Path(DATA_DIR, map_name, 'sensors', sensor_name)
    file_map = {}
    file_names = os.listdir(sensor_dir)
    for file_name in file_names:
        if 'data_dict__' not in file_name:
            continue
        file_path = Path(sensor_dir, file_name)
        data_dict = Utils.pickle_read(file_path)
        for x2 in data_dict:
            if x2 not in file_map:
                file_map[x2] = {}
            for y2 in data_dict[x2]:
                file_map[x2][y2] = file_name
    file_map_path = Path(sensor_dir, 'file_map.p')
    Utils.pickle_write(file_map_path, file_map)

# DataMap agent reads sensors saved to file and caches in RAM
# also reads in voxels data from file to determine obstacles and collisions
class DataMap(Agent.Agent):
    
    # memory_saver=False will cache all data dicts as they are read, leading to quicker fetch times but more memory
    # memory_saver=True will only cache the most recent data dict, leading to less memory but longer fetch times
    # if memory_saver=True, cache_size is number of DataMaps to keep saved in RAM as loaded from file
    def __init__(self, map_name, memory_saver=False, cache_size=8,
    roof_name='version_1', resolution=1, collision_trigger_distance=2,
    move_speed=1, target_threshold=4, collision_avoidance_threshold: float = 1.5, bounds_tolerance: float = 0,
    translational_threshold: float = 0.12, rotational_threshold: float = 0.1, discrete_space=True):
        super().__init__()
        self.map_name = map_name
        self.file_map = {} # maps x,y,z,d points to file path of data dictionary
        self.memory_saver = memory_saver # only cache the most recent data dict?
        self.cache_size = cache_size # how many data dicts to keep
        self.filepaths = [] # keep track of file path to last read data dicts for memory saver
        self.data_dicts = {} # actual data_dicts data at sensor_name, x, y, z, d
        self.roof_name = roof_name
        self.resolution = resolution
        roofs_path = Path(DATA_DIR, map_name, 'roofs', f'{roof_name}.p')
        self.set_roofs(roofs_path)
        self.grid_points = []
        self.move_speed = move_speed
        self.target_threshold = target_threshold
        self.moving = False
        self.bounds_tolerance = bounds_tolerance
        self.collision_avoidance_threshold = collision_avoidance_threshold
        self.collision_trigger_distance = collision_trigger_distance
        self.translational_threshold = translational_threshold
        self.rotational_threshold = rotational_threshold
        self.discrete_space = discrete_space



    # ******** ABSTRACT ACTION METHODS ********

    def increment_in_direction(self, point, direction, magnitude):
        x, y, z, yaw, pitch, roll = point.unpack()
        if direction == 'Forward':
            y += magnitude
        elif direction == 'Backward':
            y -= magnitude
        elif direction == 'Right':
            x += magnitude 
        elif direction == 'Left':
            x -= magnitude 
        elif direction == 'Upward':
            z += magnitude
        elif direction == 'Downward':
            z -= magnitude
        else:
            print('invalid direction')
        return DataStructure.Point(x, y, z, yaw, pitch, roll)

    # how to increment time forward during act() 
    def step(self):
        self.point = self.increment_in_direction(self.get_point(), self.get_moving_absolute(), self.move_speed)

    # move forward wrt yaw (ignore pitch and roll) -- returns destination_point
    def move_forward(self, magnitude):
        destination_point = self.increment_in_direction(self.get_point(), self.get_moving_absolute(), magnitude)
        self.moving = True
        return destination_point
    
    # strafe right wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_right(self, magnitude):
        destination_point = self.increment_in_direction(self.get_point(), self.get_moving_absolute(), magnitude)
        self.moving = True
        return destination_point

    # strafe left wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_left(self, magnitude):
        destination_point = self.increment_in_direction(self.get_point(), self.get_moving_absolute(), magnitude)
        self.moving = True
        return destination_point

    # rotate clockwise about the z-axis in place
    def rotate_clockwise(self, magnitude):
        point = self.get_point()
        yaw = point.yaw
        yaw = (yaw - magnitude) % 360
        self.point = DataStructure.Point(point.x, point.y, point.z, yaw, point.pitch, point.roll)
    
    # rotate counter-clockwise about the z-axis in place
    def rotate_counter(self, magnitude):
        point = self.get_point()
        yaw = point.yaw
        yaw = (yaw + magnitude) % 360
        self.point = DataStructure.Point(point.x, point.y, point.z, yaw, point.pitch, point.roll)
    
    # stop all motion immediately
    def stop(self):
        self.moving = False

    # check if has collided
    def check_collision(self):
        point = self.get_point()
        return self.in_object(point.x, point.y, point.z)

    # check if a collision is imminent by checking if any distance sensors are below a certain threshold
    def check_collision_avoidance(self):
        x, y, z, yaw, pitch, roll = self.get_point().unpack()
        moving_absolute = self.get_moving_absolute()
        distance_traveled = 0
        while distance_traveled < 1:#self.collision_avoidance_threshold:
            step_size = 1
            if moving_absolute == 'Forward':
                y += step_size
            elif moving_absolute == 'Backward':
                y -= step_size
            elif moving_absolute == 'Right':
                x += step_size 
            elif moving_absolute == 'Left':
                x -= step_size 
            elif moving_absolute == 'Upward':
                z += step_size
            elif moving_absolute == 'Downward':
                z -= step_size
            else:
                print('error did not move by step size')
            if self.check_outofbounds(x, y, z) or self.in_object(x, y, z):
                if moving_absolute == 'Forward':
                    self.point.y -= step_size
                elif moving_absolute == 'Backward':
                    self.point.y += step_size
                elif moving_absolute == 'Right':
                    self.point.x -= step_size 
                elif moving_absolute == 'Left':
                    self.point.x += step_size 
                elif moving_absolute == 'Upward':
                    self.point.z -= step_size
                elif moving_absolute == 'Downward':
                    self.point.z += step_size
                return True
            distance_traveled += step_size
        return False

    # check if is moving by checking if linear velocity, angular velocity, linear acceleration, or angular acceleration are above a certain threshold
    def is_moving(self, linear_threshold = 0.01, angular_threshold = 0.01, accel_threshold = 0.01):
        return self.moving

    # teleport to a point in the environment (for spawning at specific locations)
    def teleport(self, point):
        self.point = point
        self.moving = False

    # takeoff, and enter hover mode (for drones)
    def takeoff(self):
        pass


    # ******** ABSTRACT SENSOR METHODS ********

    # returns a DataStructure.Point object representing the agent's current location and orientation in the environment    
    def get_point(self):
        if self.discrete_space:
            x, y, z, yaw, pitch, roll = self.point.unpack()
            return DataStructure.Point(x, y, z, yaw, pitch, roll, discrete=True)
        return self.point

    # returns the yaw angle in Euler angles of the agent in the environment (in degrees, between 0 and 360) where 0 is along the positive-x axis, and increases clockwise
    def get_yaw(self):
        point = self.get_point()
        return point.yaw

    # get image from drone's camera, camera_name is the name of the camera to capture from, which has predefined settings
    def get_image(self, camera_name):

        image = self.get_data_point(self.get_point(), camera_name)

        return image

    # return dictionary of various meta data saved in local meta.json file
    def get_sensor_meta(self, sensor_name):
        sensor_dir = Path(DATA_DIR, self.map_name, 'sensors', sensor_name)
        sensor_meta_path = Path(sensor_dir, 'sensor_meta.json')
        sensor_meta = Utils.json_read(sensor_meta_path)
        return sensor_meta


    # ******** BASE HELPER METHODS ********

    # clear up memory
        # sensor_name=None will clear all memory, otherwise only clear mem associated with sensor_name
    def clear_cache(self, sensor_name=None):
        self.fetched_data.clear()
        if sensor_name is None:
            self.data_dicts.clear()
        else:
            self.data_dicts[sensor_name].clear()
            del self.data_dicts[sensor_name] 

    # map file paths of data dictionaries to [sensor_name, x, y] keys
        # NOTE: this is an efficiency choice that considers DataMaps are made by chunking only x and y and not z and d
    def get_filepath(self, sensor_name, x, y):
        sensor_dir = Path(DATA_DIR, self.map_name, 'sensors', sensor_name)
        if sensor_name not in self.file_map:
            self.file_map[sensor_name] = {}
            # get meta data from file -- linking each x,y coordinate to data_dict part
            file_map_path = Path(sensor_dir, 'file_map.p')
            if not os.path.exists(file_map_path):
                write_file_map(self.map_name, sensor_name)
            file_map = Utils.pickle_read(file_map_path)
            self.file_map[sensor_name] = file_map
        exists = sensor_name in self.file_map and x in self.file_map[sensor_name] and y in self.file_map[sensor_name][x]
        if exists:
            return Path(sensor_dir, self.file_map[sensor_name][x][y])
        else:
            return None

    # gets observations from given data dict if available
    def get_observation(self, x, y, z, yaw, data_dict):
        if x in data_dict and y in data_dict[x] and z in data_dict[x][y] and yaw in data_dict[x][y][z]:
            return data_dict[x][y][z][yaw]
        else:
            return None
        
    # reads corresponding data_dict_part into memory if single data point does not exist in current data_dict
    # if (self.memory_saver or use_memory_saver) then does not store data_dict_part into memory
    def get_data_point(self, point, sensor_name, use_memory_saver=False):
        observation = None  
        x, y, z, yaw, pitch, roll = point.unpack()
        filepath = self.get_filepath(sensor_name, x, y)
        if filepath is not None and filepath not in self.data_dicts:
            self.filepaths.append(filepath)
            if (self.memory_saver or use_memory_saver) and len(self.filepaths) > self.cache_size:
                remove_filepath = self.filepaths[0]
                del self.data_dicts[remove_filepath]
                del self.filepaths[0]
            self.data_dicts[filepath] = Utils.pickle_read(filepath)
        if filepath is not None:
            observation = self.get_observation(x, y, z, yaw, self.data_dicts[filepath])
        return observation
    
    # reads and sets roofs object from file
    def set_roofs(self, roofs_path):
    
        # read roofs from file
        self.roofs_path = roofs_path
        self.roofs = Utils.pickle_read(roofs_path)
        
        # set x stats
        xs = list(self.roofs.keys())
        self.x_min = np.min(xs) # inclusive
        self.x_max = np.max(xs) + self.resolution # exclusive
        self.x_n = int((self.x_max - self.x_min) / self.resolution)
        
        # set y stats
        self.y_min, self.y_max = 99999, -99999
        for x in xs:
            ys = list(self.roofs[x].keys())
            self.y_min = min(self.y_min, min(ys))
            self.y_max = max(self.y_max, max(ys))
        self.y_max += self.resolution
        self.y_n = int((self.y_max - self.y_min) / self.resolution)

        # make numpy array of roofs
        self.roofs_array = np.full((self.x_n, self.y_n), np.nan)
        for xi in range(self.x_n):
            x = self.x_min + xi*self.resolution
            if x not in self.roofs:
                continue
            for yi in range(self.y_n):
                y = self.y_min + yi*self.resolution
                if y not in self.roofs[x]:
                    continue
                self.roofs_array[xi, yi] = self.roofs[x][y]

        # set array shifts to translate between dictionary and array of positions
        self.x_shift = -1*self.x_min
        self.y_shift = -1*self.y_min
    
    # checks if position is in object based on collision_trigger_distance (meters) above rooftop
    def in_object(self, x, y, z):
        return z <= self.get_roof(x, y) + self.collision_trigger_distance

    def get_roof(self, x, y):
        return self.roofs[x][y]