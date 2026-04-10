from OmniNaviPy.modules import Environment
import numpy as np

# helper functions
def descrite_yaw(yaw):
    if yaw is not None:
        if yaw  < 45 or yaw >= 315:
            yaw = 0
        elif yaw >= 45 and yaw < 135:
            yaw = 90
        elif yaw  >= 135 and yaw < 225:
            yaw = 180
        elif yaw  >= 225 and yaw < 315:
            yaw = 270
    return yaw

# keep track of agent's position and orientation in given environment 
class Point:
    # x, y, z are in meters following Euclidean space, yaw is in degrees following Euler space between 0 and 360 where 0 is along the positive-x axis and increases clockwise
    # if discrete=True, then point is in in discrete space (1 meter, 90 degree increments)
    def __init__(self, x: float, y: float, z: float, yaw: float=0, pitch: float=0, roll: float=0, discrete: bool = False):
        self.discrete = discrete
        if self.discrete:
            self.x = round(x)
            self.y = round(y)
            self.z = round(z)
            self.yaw = descrite_yaw(yaw)
            self.pitch = descrite_yaw(pitch)
            self.roll = descrite_yaw(roll)
        else:
            self.x = x
            self.y = y
            self.z = z
            self.yaw = yaw
            self.pitch = pitch
            self.roll = roll
    
    # xyz=True will calculate distance in 3D space, xyz=False will calculate distance in 2D space ignoring z value
    def distance(self, point, xyz=True):
        return np.linalg.norm(self.displacement(point, xyz))

    def displacement(self, point, xyz=True):
        if xyz:
            return self.numpy()[:3] - point.numpy()[:3]
        return self.numpy()[:2] - point.numpy()[:2]

    def numpy(self):
        return np.array([self.x , self.y, self.z, self.yaw, self.pitch, self.roll])

    def unpack(self):
        return self.x, self.y, self.z, self.yaw, self.pitch, self.roll
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.yaw == other.yaw and self.pitch == other.pitch and self.roll == other.roll

    def __str__(self):
        x, y, z, yaw, pitch, roll = self.unpack()
        s = f'x:{x} y:{y} z:{z} yaw:{yaw} pitch:{pitch} roll:{roll}'
        return s

    def __repr__(self):
        return str(self)
        

    # returns integer value indicating direction robot is facing, given yaw
    def get_direction(self):
        yaw = self.yaw
        if yaw  < 45 or yaw >= 315:
            direction = 1
        elif yaw >= 45 and yaw < 135:
            direction = 0
        elif yaw  >= 135 and yaw < 225:
            direction = 3
        elif yaw  >= 225 and yaw < 315:
            direction = 2
        return direction