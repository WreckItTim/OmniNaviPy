from OmniNaviPy.modules import DataStructure
from OmniNaviPy.modules import Agent
from OmniNaviPy.modules import Utils
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import subprocess
import platform
import airsim
import psutil
import math
import time



# child classes of Agent to work with Microsoft AirSim


# **** HELPER METHODS****

# flip airsim's drone coordinates to euclidean coordinates or vice versa
# airsim coordinates = +x forward, +y right, +z down
# euclidean coordinates = +x right, +y forward, +z up
def euclidean_to_airsim(x, y, z):
    return y, x, -1*z
def airsim_to_euclidean(x, y, z):
    return y, x, -1*z

# flip airsim's drone yaw to euler angles or vice versa
# airsim yaw = 0 radians is facing forward, pi/2 radians is facing right, pi radians is facing backward, -pi/2 radians is facing left
# euler yaw = 0 degrees is facing right, 90 degrees is facing forward, 180 degrees is facing left, 270 degrees is facing backward
def euler_to_airsim(yaw):
    if yaw <= 90:
        yaw = math.radians(90 - yaw)
    elif yaw < 270:
        yaw = -1*math.radians(yaw - 90)
    else:
        yaw = math.radians(450 - yaw)
    return yaw
def airsim_to_euler(yaw):
    yaw = math.degrees(yaw)
    if yaw <= 90:
        yaw = 90 - yaw
    else:
        yaw = 450 - yaw
    return yaw

# **** AIRSIM AGENT CLASS ****
# Parent class to launch and connect to AirSim (default values are experimentally found to be the most stable)
    # release_path is the path to the AirSim release, if None it is assumed AirSim is already running
    # settings_name is the name of the settings file to use, which should be located in the AirSim release directory
    # flags is a list of strings that are passed as command line arguments to the AirSim executable when launching, for example ['-windowed']
    # timeout is the number of seconds to wait for AirSim to finish a command before raising an error
    # render_animals is a boolean that determines whether to render animals in the simulator, which can be a significant source of visual noise and is not necessary for navigation tasks
    # render_foilage is a boolean that determines whether to render foilage in the simulator, which can be uncollidable trees in the background
    # weather_type is an integer that determines the type of weather to render -- -1 for no weather, 0 for rain, 1 for snow, 2 for fog
    # weather_degree is an integer that determines the degree of weather to render, which can be a value from 0 to 100
    # additional_settings is a dictionary that can be used to overwrite any settings in the settings file, for example {'CameraDefaults': {'FOV_Degrees': 90}}
    # move_speed is the average speed at which the drone moves as controlled by the PID flight controller
    # navigation_frequency is the frequency at which to check on the progress of asynchronous actions and step through the simulator
    # clock_speed is the speed at which time passes in the simulator, where 1 is real-time, 2 is twice as fast as real-time, etc
    # image_type is the type of image to capture from the drone's camera, where 0 is scene, 2 is depth, see AirSim docs for more
    # camera_height is the height of the camera image to capture
    # camera_width is the width of the camera image to capture
    # max_distance is the maximum distance that the drone's distance sensors can detect, which determines the threshold for collision avoidance
    # draw_debug_points is a boolean that determines whether to draw green cubes in Airsim to show distance sensors
    # translational_threshold is the distance in meters that the drone must move to be considered as having moved for the purpose of determining when an action is done
    # rotational_threshold is the angle in degrees that the drone must rotate to be considered as having rotated for the purpose of determining when an action is done
    # target_threshold is the distance in meters to the target point that is considered as having reached the target for the purpose of determining when an episode is done
class MicrosoftAirSim(Agent.Agent):

    def __init__(self, release_path: str = None, flags: list = [],
    timeout: int = 60, render_animals: bool = False, render_foilage: bool = True, weather_type: int = -1, 
    weather_degree: int = 1, additional_settings: dict = {}, move_speed: float = 1, 
    navigation_frequency: float = 0.1, clock_speed: float = 1, image_type: int = 2, camera_height: int = 144,
    camera_width: int = 256, max_distance: int = 100, draw_debug_points: bool = False,
    translational_threshold: float = 0.12, rotational_threshold: float = 0.1, target_threshold: float = 4,
    collision_threshold: float = 1.5, bounds_tolerance: float = 1.5, fixed_z: float = None, discrete_space: bool = False):
        super().__init__()
        self.release_path = release_path
        self.flags = flags
        self.timeout = timeout
        self.render_animals = render_animals
        self.render_foilage = render_foilage
        self.weather_type = weather_type
        self.weather_degree = weather_degree
        self.clock_speed = clock_speed
        self.move_speed = move_speed
        self.navigation_frequency = navigation_frequency
        self.image_type = image_type
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_distance = max_distance
        self.draw_debug_points = draw_debug_points
        self.translational_threshold = translational_threshold
        self.rotational_threshold = rotational_threshold
        self.target_threshold = target_threshold
        self.collision_threshold = collision_threshold
        self.bounds_tolerance = bounds_tolerance
        self.fixed_z = fixed_z
        self.discrete_space = discrete_space
        self.make_settings(additional_settings) # create settings file with specified settings and save path to self.settings_path
        print('release_path', release_path)
        if release_path is not None:
            self.launch() # launch and airsim instance and connect to it
        else:
            self.connect() # connect to an already launched airsim instance


    # ******** ACTION METHODS ********

    # how to increment time forward during act() 
    def step(self):

        # we will just sleep for a short duration, since AirSim actions are asynchronous and handled by the native client's PID flight controller
        time.sleep(self.navigation_frequency) # honk sshhhhh

    # move forward wrt yaw (ignore pitch and roll) -- returns destination_point
    def move_forward(self, magnitude, asynchronous=True):
        
        # get orientation to determine translational vector
        yaw = math.radians(self.get_yaw())
        
        # asynchronously move forward, which we will check on and step through in the step() function
        return self.move(magnitude*math.cos(yaw), magnitude*math.sin(yaw), 0, speed=self.move_speed, join=not asynchronous)
    
    # strafe right wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_right(self, magnitude, asynchronous=True):
        
        # get orientation to determine translational vector
        yaw = math.radians(self.get_yaw())
        
        # asynchronously move right, which we will check on and step through in the step() function
        return self.move(magnitude*math.cos(yaw-math.pi/2), magnitude*math.sin(yaw-math.pi/2), 0, speed=self.move_speed, join=not asynchronous)
    
    # strafe left wrt yaw (ignore pitch and roll) -- returns destination_point
    def strafe_left(self, magnitude, asynchronous=True):
        
        # get orientation to determine translational vector
        yaw = math.radians(self.get_yaw())
        
        # asynchronously move left, which we will check on and step through in the step() function
        return self.move(magnitude*math.cos(yaw+math.pi/2), magnitude*math.sin(yaw+math.pi/2), 0, speed=self.move_speed, join=not asynchronous)

    # rotate clockwise about the z-axis in place
    def rotate_clockwise(self, magnitude, asynchronous=False):

        # synchronously rotate clockwise, since airsim will pass the desired angle then oscilate back until motion dies down
        self.rotate(magnitude, threshold=self.rotational_threshold, join=not asynchronous)
    
    # rotate counter-clockwise about the z-axis in place
    def rotate_counter(self, magnitude, asynchronous=False):

        # synchronously rotate counter-clockwise, since airsim will pass the desired angle then oscilate back until motion dies down
        self.rotate(-magnitude, threshold=self.rotational_threshold, join=not asynchronous)
    
    # stop all motion immediately
    def stop(self):

        # airsim does not have a stop command, but we can achieve this by sending a move command with zero velocity and zero yaw rate
        self.client.rotateByYawRateAsync(0, 1).join()
        self.client.moveByVelocityAsync(0, 0, 0, 1).join()

    # check if has collided
    def check_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided
        return has_collided 

    # check if a collision is imminent by checking if any distance sensors are below a certain threshold
    def check_collision_avoidance(self):
        return self.nearest_distance_sensor() < self.collision_threshold

    # check if is moving by checking if linear velocity, angular velocity, linear acceleration, or angular acceleration are above a certain threshold
    def is_moving(self, linear_threshold = 0.01, angular_threshold = 0.01, accel_threshold = 0.01):
        state = self.client.getMultirotorState()
        lin_vel = state.kinematics_estimated.linear_velocity
        ang_vel = state.kinematics_estimated.angular_velocity
        linear_speed = math.sqrt(lin_vel.x_val**2 + lin_vel.y_val**2 + lin_vel.z_val**2)
        angular_speed = math.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)
        is_translating = linear_speed > linear_threshold
        is_rotating = angular_speed > angular_threshold
        linear_accel = state.kinematics_estimated.linear_acceleration
        linear_accel = math.sqrt(linear_accel.x_val**2 + linear_accel.y_val**2 + linear_accel.z_val**2)
        angular_accel = state.kinematics_estimated.angular_acceleration
        angular_accel = math.sqrt(angular_accel.x_val**2 + angular_accel.y_val**2 + angular_accel.z_val**2)
        is_accelerating = linear_accel > accel_threshold or angular_accel > accel_threshold
        return is_translating or is_rotating or is_accelerating

    # moves drone directly to given pose and orientation, ignoring physics and collisions -- useful for spawning at specific locations
    def teleport(self, point, roll=0, pitch=0, ignore_collision=True, stabelize=True):

        # change euclidean coords x, y, z to airsim drone coords y, x, -z
        x, y, z = euclidean_to_airsim(point.x, point.y, point.z)

        #  change euler yaw to airsim yaw radians
        yaw = euler_to_airsim(point.yaw)

        # create airsim Pose object with position and orientation
        pose = airsim.Pose(
            airsim.Vector3r(x, y, z), 
            airsim.to_quaternion(pitch, roll, yaw)
        )

        # directly set to new pose object
        self.client.simSetVehiclePose(pose, ignore_collision=ignore_collision)
        
        # stabalize drone?
        if stabelize:
            self.stabelize()
            
    def take_off(self):
        self.client.takeoffAsync(timeout_sec = self.timeout).join()


    # ******** SENSOR METHODS ********

    # returns a DataStructure.Point object representing the agent's current location and orientation in the simulator    
    def get_point(self):

        # query airsim client for estimated position
        x, y, z = self.get_position()

        # query airsim for estimated yaw
        yaw = self.get_yaw()

        return DataStructure.Point(x, y, z, yaw)

    # get rotation about the z-axis (yaw) in euler angles (0 degrees facing right, 90 degrees facing forward, 180 degrees facing left, 270 degrees facing backward) or airsim radians (0 radians facing forward, pi/2 radians facing right, pi radians facing backward, -pi/2 radians facing left)
    def get_yaw(self, as_euler=True):

        # querty airsim for esimated quaternions
        q = self.client.getMultirotorState().kinematics_estimated.orientation
        
        # convert quaternions to eularian angles
        pitch, roll, yaw = airsim.to_eularian_angles(q)

        # convert airsim yaw radians to normal euler degrees
        if as_euler:
            yaw = airsim_to_euler(yaw)
        
        return yaw

    # get image from drone's camera, camera_name is the name of the camera to capture from, which has predefined settings
    def get_image(self, camera_name):

        # forward facing depth (default)
        if camera_name in ['DepthV1']:
            image = self.camera(camera_view='0', image_type=2)

        return image

    # get metadata about the sensor, such as the shape of the observation space
    def get_sensor_meta(self, camera_name):
        if camera_name in ['DepthV1']:
            sensor_meta = {
                'shape': (1, self.camera_height, self.camera_width)
            }

        return sensor_meta

    # **** AIRSIM DRONE HELPERS ****

    # camera_view values:
        # 'front_center' or '0'
        # 'front_'Right' or '1'
        # 'front_left' or '2'
        # 'bottom_center' or '3'
        # 'back_center' or '4'
    # image_type values:
        # Scene = 0, 
        # DepthPlanar = 1, 
        # DepthPerspective = 2, >>> use this for depth maps
        # DepthVis = 3, 
        # DisparityNormalized = 4,
        # Segmentation = 5,
        # SurfaceNormals = 6,
        # Infrared = 7,
        # OpticalFlow = 8,
        # OpticalFlowVis = 9
    def camera(self, camera_view='0', image_type=2, compress=False, view_img=False, 
                    out_dir=None, make_channel_first=True
                ):
        #print('AIR-CMD camera')
        if image_type in [1, 2, 3, 4]:
            as_float = True
            is_gray = True
            is_image = False
        else:
            as_float = False
            is_gray = False
            is_image = True
        image_request = airsim.ImageRequest(camera_view, image_type, as_float, compress)
        img_array = []
        while len(img_array) <= 0: # loop for dead images (happens some times)
            response = self.client.simGetImages([image_request])[0]
            if as_float:
                np_flat = np.array(response.image_data_float, dtype=float)
            else:
                np_flat = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            if is_gray:
                img_array = np.reshape(np_flat, (response.height, response.width))
            else:
                img_array = np.reshape(np_flat, (response.height, response.width, 3))
                    
        if view_img:
            if is_gray:
                plt.imshow(img_array, cmap='grey', vmin=0, vmax=255)
            else:
                plt.imshow(img_array)
            plt.show()
            
        if is_gray:
            img_array[img_array<1] = 1
            img_array[img_array>255] = 255
            img_array = img_array.astype(np.uint8)

        # make channel-first
        if make_channel_first and len(img_array) > 0:
            if is_gray:
                img_array = np.expand_dims(img_array, axis=0)
            else:
                img_array = np.moveaxis(img_array, 2, 0)
                    
        return img_array

    # check distance sensors in the direction of movement to get nearest obstacle distance (in direction moving)
    def nearest_distance_sensor(self):
        moving_relative = self.get_moving_relative()
        return min(
            self.client.getDistanceSensorData(distance_sensor_name=f"Distance{moving_relative}1").distance,
            self.client.getDistanceSensorData(distance_sensor_name=f"Distance{moving_relative}2").distance,
            self.client.getDistanceSensorData(distance_sensor_name=f"Distance{moving_relative}3").distance,
            )

    # moves relative magnitudes to current position
    def move(self, x_rel, y_rel, z_rel, speed=2, stabelize=True, join=True):

        # get current position to calculate target position
        x, y, z = self.get_position(as_euclidean=False)
        current_position = np.array([x, y, z])

        # flip relative movement from euclidean to airsim coords and calculate target position, adjusting for fixed altitude if necessary
        x_rel, y_rel, z_rel = euclidean_to_airsim(x_rel, y_rel, z_rel)
        destination_position = current_position + np.array([x_rel, y_rel, z_rel])
        if self.fixed_z is not None:
            destination_position[2] = -1*self.fixed_z # negate fixed_z since airsim z is negative up
        if self.discrete_space:
            destination_position[0] = round(destination_position[0])
            destination_position[1] = round(destination_position[1])
            destination_position[2] = round(destination_position[2])
        
        # asynchronously move to target position, which we will check progress using the step() function unless join=True
        thread = self.client.moveToPositionAsync(destination_position[0], destination_position[1], destination_position[2], 
                                            speed, timeout_sec = self.timeout)

        # synchronous move
        if join:
            thread.join()
            # stabalize drone?
            if stabelize:
                self.stabelize()

        return DataStructure.Point(destination_position[0], destination_position[1], destination_position[2])

    # rotate relative to current yaw, rel_yaw is in degrees, positive for clockwise, negative for counter-clockwise
    def rotate(self, rel_yaw: float, threshold: float = 0.1, stabelize: bool = True, join: bool = True):
        # get current yaw to calculate target yaw
        q = self.client.getMultirotorState().kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        yaw = np.degrees(yaw)

        # calculate target yaw
        target_yaw = yaw + rel_yaw
        if self.discrete_space:
            target_yaw = round(target_yaw / 90) * 90
            
        # asynchronously rotate to target yaw, which we will check progress on using the step() function unless join=True
        thread = self.client.rotateToYawAsync(target_yaw, margin = threshold, timeout_sec = self.timeout)

        # synchronous rotate
        if join:
            thread.join()
            # adjust for altitude changes
            if self.fixed_z is not None:
                x, y, z = self.get_position(as_euclidean=False)
                self.client.moveToPositionAsync(x, y, -1*self.fixed_z, self.move_speed).join()
            # stabalize drone?
            if stabelize:
                self.stabelize()

    # stabelize method is a stop_gap to fix AirSim's y-drift problem
    # see this GitHub ticket, with youtube video showing problem:
    # https://github.com/microsoft/AirSim/issues/4780
    def stabelize(self):
        self.client.rotateByYawRateAsync(0, 0.001).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.001).join()

    # get position in euclidean coordinates (x right, y forward, z up) or airsim coordinates (x forward, y right, z down)
    def get_position(self, as_euclidean=True):

        # query airsim for estimated position
        pos = self.client.getMultirotorState().kinematics_estimated.position
        
        # change drone coords y, x, -z to euclidean coords x, y, z 
        if as_euclidean:
            x, y, z = airsim_to_euclidean(pos.x_val, pos.y_val, pos.z_val)
        else:
            x, y, z = pos.x_val, pos.y_val, pos.z_val

        return x, y, z

    # remove all animals in the simulator, which can be a significant source of visual noise and is not necessary for navigation tasks
    def remove_all_animals(self):
        objs = self.client.simListSceneObjects()
        animals = [name for name in objs if 'Deer' in name or 'Raccoon' in name or 'Animal' in name]
        _ = [self.client.simDestroyObject(name) for name in animals]

    # remove all foilage in the simulator, which can be uncollidable trees in the background that can confuse segmentation
    def remove_all_foilage(self):
        objs = self.client.simListSceneObjects()
        foliages = [name for name in objs if 'Foliage' in name]
        _ = [self.client.simDestroyObject(name) for name in foliages] 

    # clear all weather effects in the simulator
    def clear_weather(self):
        for i in range(8):
            self.client.simSetWeatherParameter(i, 0)

    # set weather effects in the simulator, weather_type is an integer that determines the type of weather to render -- -1 for no weather, 0 for rain, 1 for snow, 2 for fog, weather_degree is an integer that determines the degree of weather to render, which can be a value from 0 to 100
    def set_weather(self, weather_type: int, weather_degree: int):
        self.client.simEnableWeather(True)
        self.client.simSetWeatherParameter(weather_type, weather_degree)
        # add wet roads with rain
        if self.weather_type == 0:
            self.client.simSetWeatherParameter(1, self.weather_degree)
        # add snowy roads with snow
        if self.weather_type == 2:
            self.client.simSetWeatherParameter(3, self.weather_degree)
        # add leafy roads with leafs
        if self.weather_type == 4:
            self.client.simSetWeatherParameter(5, self.weather_degree)

    # connect to existing AirSim instance for feedback loop, for example if connection is lost or if AirSim is launched after this class is initialized
    def connect(self):
        # establish communication link with airsim client
        self.client = airsim.MultirotorClient(
            timeout_value=5*60, # if no communication in this time is made then will throw TimeoutError
        )
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(1)
        self.client.takeoffAsync().join()
        time.sleep(1)

    # launch airsim map from given OS
    def launch(self):
        # set flags
        flags = ''
        if self.flags is not None:
            flags = ' '.join(self.flags)
        
        # launch AirSim release from OS
        os_name = platform.system()
        prefix = '' if os_name == 'Windows' else 'sh '
        terminal_command = f'{prefix}{self.release_path} {flags} -settings=\"{self.settings_path}\"  > /dev/null 2>&1' # silence annoying airsim output
        print('issuing command:', terminal_command)
        if os_name == 'Windows':
            process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif os_name == 'Linux':
            process = subprocess.Popen(terminal_command, shell=True, start_new_session=True)
        self.pid = process.pid
            
        # wait for map to load
        time.sleep(20)

        self.connect()

        # little fun critters that Microsoft added who run around and can get in the way of things
        if not self.render_animals:
            self.remove_all_animals() # PETA has joined the chat
        # the background trees/bushes in the horizon is labeled as Foilage and can confuse segmentation
        if not self.render_foilage:
            self.remove_all_foilage() # USFS has joined the chat
        # set weather
        if self.weather_type > -1:
            self.set_weather(self.weather_type, self.weather_degree)
        # wait to render
        time.sleep(2)
        
    # clean up loaded airsim resources
    def close(self):
        # this should keep child in tact to kill same process created (can handle multi in parallel)
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass

    # initialize json file for AirSim to read-in at launch, which specifies settings for the simulator such as weather, rendering, and sensors -- this is necessary to launch AirSim with the correct settings
    def make_settings(self, addtional_settings={}):

        settings = {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
            "SettingsVersion": 1.2,

            "SimMode": "Multirotor",

            "ViewMode": "SpringArmChase",

            "EngineSound": False,
            "LogMessagesVisible": False,

            "SubWindows": [
            ],

            'ClockSpeed': self.clock_speed, # speed up, >1, or slow down, <1. For aisrim, generally don't go higher than 10 - but this is dependent on your setup
            "CameraDefaults": {
                "CaptureSettings": [ {
                    "Width": self.camera_width,
                    "Height": self.camera_height,
                    "FOV_Degrees": 90,
                    "AutoExposureSpeed": 100,
                    "MotionBlurAmount": 0
                    }
                ]
            },
            "Vehicles": {
                "SimpleFlight": {
                    "VehicleType": "SimpleFlight",
                    "EnableCollisionPassthrogh": False,
                    "Sensors": {
                        "DistanceForward1": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0.5, "Y": 0, "Z": 0,
                            "Yaw": 0, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceForward2": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0.5, "Y": -0.5, "Z": 0,
                            "Yaw": 0, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceForward3": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0.5, "Y": 0.5, "Z": 0,
                            "Yaw": 0, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceLeft1": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0, "Y": -0.5, "Z": 0,
                            "Yaw": -90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceLeft2": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0.5, "Y": -0.5, "Z": 0,
                            "Yaw": -90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceLeft3": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": -0.5, "Y": -0.5, "Z": 0,
                            "Yaw": -90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceRight1": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0, "Y": 0.5, "Z": 0,
                            "Yaw": 90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceRight2": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": 0.5, "Y": 0.5, "Z": 0,
                            "Yaw": 90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                        "DistanceRight3": {
                            "SensorType": 5,
                            "Enabled": True,
                            "X": -0.5, "Y": 0.5, "Z": 0,
                            "Yaw": 90, "Pitch": 0, "Roll": 0,
                            'MaxDistance':self.max_distance,
                            "DrawDebugPoints": self.draw_debug_points
                        },
                    }
                }
            }
        }

        # add additional settings
        settings.update(addtional_settings)

        # temporary write to file to send path to AirSim at launch, since AirSim only accepts settings as a file path string and not as a dictionary or other data structure
        repository_dir = Utils.get_global('repository_directory')
        self.settings_path = Path(repository_dir, 'local', 'temp_airsim_settings.json')
        Utils.json_write(self.settings_path, settings)
        

    # ******** OTHER FUNCTIONS ********

    # writes a 3D voxel grid to file -- captures from AirSim using native Unreal Engine surfaces
    # center is the center of the voxel grid in euclidean coordinates
    # super_cube_res is the resolution of the super cube that defines the overall bounds of the voxel grid
    # sub_cube_res is the resolution of the sub cubes that define the individual voxels
    # output_path is the path to save the voxel grid to
    def write_voxels(self, center: tuple, super_cube_res: int, sub_cube_res: int, output_path: str):
        center = airsim.Vector3r(*center)
        self.client.simCreateVoxelGrid(center, super_cube_res, super_cube_res, super_cube_res, sub_cube_res, output_path)