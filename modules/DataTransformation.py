from pathlib import Path
import numpy as np
import os

# default assumptions for all color images (unless transformation assumes otherwise):
    # dimensions are: [channel, height, width]
    # three channels are RGB

# default assumptions for all grey-scale images (unless transformation assumes otherwise):
    # dimensions are: [1, height, width]

# parent class
# must define transform() that inputs observation and outputs transfomred observation 
class DataTransformation:
    def transform(self, observation):
        raise NotImplementedError

# applies a given pipeline of many transformations
class Pipeline(DataTransformation):
    def __init__(self, data_transformations):
        self.data_transformations = data_transformations

    def transform(self, observation):
        for data_transformation in self.data_transformations:
            observation = data_transformation.transform(observation)
        return observation

# normalizes to given ranges
class Normalize(DataTransformation):
    def __init__(self, 
				 min_input=0,
				 max_input=255, # default horizon for pixel values in image
				 min_output=0.1, # reserve 0 for missing or erroneous data
				 max_output=1,
				 left = None, # value to set if below min_input (otherwise defaults to min_input)
				 right = None, # value to set if above max_input (otherwise defaults to max_input)
                ):
        self.min_input = min_input
        self.max_input = max_input
        self.min_output = min_output
        self.max_output = max_output
        self.left = left
        self.right = right

    def transform(self, observation):
        observation = np.interp(observation,
                                (self.min_input, self.max_input),
                                (self.min_output, self.max_output),
                                left=self.left, right=self.right,
                                )
        return observation

# converts to old format that used direction
# 0 forward, 1 right, 2 down, 3 left
class ConvertToDirection(DataTransformation):
    def transform(self, observation):
        yaw = observation[0] # assuming yaw is the first component of the observation vector, can adjust if not the case
        if yaw  < 45 or yaw >= 315:
            return 1
        elif yaw >= 45 and yaw < 135:
            return 0
        elif yaw  >= 135 and yaw < 225:
            return 3
        elif yaw  >= 225 and yaw < 315:
            return 2
        else:
            raise ValueError(f'Invalid yaw value: {yaw}, expected one of [0, 90, 180, 270]')    


#### misc functions

# takes either an rgb or bgr image as input then switches to the other
# assumes image is channel first
def switch_bgr_and_rgb(img, copy_data=True):
    return img[::-1, :, :]


# takes an image as input and switches between channel first and last
    # i.e., [rgb, height, width] to [height, width, rgb]
def channel_first_to_last(img):
    return np.moveaxis(img, 0, 2)
def channel_last_to_first(img):
    return np.moveaxis(img, 2, 0)


# takes a numpy array as input and removes single-dimension axis
def squeeze_array(arr):
    return np.squeeze(arr)