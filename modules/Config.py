from OmniNaviPy.modules import Utils
from OmniNaviPy.modules import Agent
import numpy as np

# discrete action space and discrete obsevation space
# forward facing depth maps
# move forward, strafe left, strafe right, rotate counter, rotate clockwise -- all at fixed altitude
def beta(agent:Agent.Agent, depth_sensor_name='DepthV1', steps_multiplier = 8, goal_tolerance=4):

    # adjust agents movements for discrete space and fixed altitude
    agent.set_discrete_space(True)
    agent.set_fixed_z(4)

    # action space
    from OmniNaviPy.modules import Action
    from OmniNaviPy.modules import Actor
    actions = []
    actions.append(Action.RotateClockwise(agent, 90))
    actions.append(Action.RotateCounter(agent, 90))
    action_magnitudes = [1, 2, 4, 8, 16, 32]
    for magnitude in action_magnitudes:
        actions.append(Action.StrafeRight(agent, magnitude))
    for magnitude in action_magnitudes:
        actions.append(Action.StrafeLeft(agent, magnitude))
    for magnitude in action_magnitudes:
        actions.append(Action.Forward(agent, magnitude))
    actor = Actor.Discrete(actions)

    # observation space
    from OmniNaviPy.modules import DataTransformation
    from OmniNaviPy.modules import Observer
    from OmniNaviPy.modules import Sensor
    img_observer = Observer.Observer([
            Sensor.Camera(agent, depth_sensor_name),
        ], n_history=3, data_type=np.uint8)
    # convert yaw to direction then normalize (using old coordinate system that config beta model was trained in)
    old_yaw_pipeline = DataTransformation.Pipeline([
        DataTransformation.ConvertToDirection(),
        DataTransformation.Normalize(max_input=3),
    ])
    vec_observer = Observer.Observer([
            Sensor.RelativeGoal(agent, xyz=False),
            Sensor.DistanceBounds(agent, DataTransformation.Normalize(max_input=255)),
            Sensor.CurrentYaw(agent, old_yaw_pipeline),
        ], n_history=3)
    observer_dict = {
        'vec':vec_observer,
        'img':img_observer,
    }
    observer = Observer.DictObserver(observer_dict)

    # termination criteria
    from OmniNaviPy.modules import Terminator
    terminators = [
        Terminator.Goal(agent, goal_tolerance),
        Terminator.MaxSteps(steps_multiplier),
    ]

    return actor, observer, terminators