
import copy

class Episode:
    def __init__(self, start_point, target_point=None, ground_truth_trajectory=None, save_observations=False):
        self.start_point = start_point
        self.target_point = target_point
        self.ground_truth_trajectory = ground_truth_trajectory
        self.save_observations = save_observations
        self.waypoint_history = []
        self.path_history = []
        self.observations_history = []
        self.action_history = []
        self.point = start_point
        self.observations = None
        self.termination = None
        self.action = None
        self.waypoint = None
        # states can hold arbitrary information as seen fit
        self.state = {}
        self.states = [self.state]

    def new_step(self):
        self.state = {}
        self.states.append(self.state)

    def add_to_state(self, key, value):
        self.state[key] = value

    def add_action(self, action_dict):
        self.action = action_dict
        self.action_history.append(action_dict)

    def get_depth_map(self):
        if self.observations is not None and 'img' in self.observations:
            return self.observations['img'][0]
        else:
            return None

    def add_termination(self, termination):
        self.termination = termination

    def add_waypoint(self, waypoint):
        self.waypoint_history.append(waypoint)
        self.waypoint = waypoint

    def add_point(self, point):
        self.path_history.append(point)
        self.point = point

    def add_observations(self, observations):
        if self.save_observations:
            self.observations_history.append(copy.deepcopy(observations))
        self.observations = observations

    def n_steps(self):
        return len(self.action_history)