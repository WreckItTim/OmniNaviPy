from OmniNaviPy.modules import DataStructure
from OmniNaviPy.modules import Utils

# trajectory class storesa list of points representing a path through the environment, along with any relevant metadata 
class Trajectory:
    def __init__(self, points:list[DataStructure.Point], difficulty=None):
        self.points = points
        self.difficulty = difficulty

    def n_steps(self):
        return len(self.points)-1

    def reached_goal(self):
        return self.points[-1].is_goal

# reads in dictionary of diffulty indexed trajectories, and processes as requested
# difficulties=None will read all difficulties from file, otherwise expects a list of difficulty keys
# n_per_difficulty=None will read all trajectories from file, otherwise an integer value specifying number of trajectories to read PER DIFFICULTY
# as_list=False will return trajectories as {difficulty: list of Trajectory objects}, as_list=True will return trajectories as a single list of Trajectory objects (ignoring difficulty keys)
def read_curriculum(curriculum_path, difficulties=None, n_per_difficulty=None, as_list=False):
    trajectories_dict = Utils.pickle_read(curriculum_path)
    for difficulty in list(trajectories_dict.keys()):
        if difficulties is not None and difficulty not in difficulties:
            del trajectories_dict[difficulty]
            continue
        if n_per_difficulty is not None:
            trajectories_dict[difficulty] = trajectories_dict[difficulty][:n_per_difficulty]
    if as_list:
        trajectories = []
        for difficulty in trajectories_dict:
            trajectories.extend(trajectories_dict[difficulty])
        return trajectories
    return trajectories_dict