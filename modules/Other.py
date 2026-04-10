from OmniNaviPy.modules import DataStructure
from OmniNaviPy.modules import Component
from OmniNaviPy.modules import Episode
from OmniNaviPy.modules import World
import numpy as np
import os
try:
    import ollama
except ModuleNotFoundError:
    pass
from matplotlib.figure import Figure 
from matplotlib.backends.backend_agg import FigureCanvasAgg

# misc operations to execute on start/step/end
class Other(Component.Component):
    pass

# # evaluates at set intervals during training and saves model weights if improved
# class LearningCurve(Other):
    
#     def __init__(self, evaluation_episodes, model, eval_frequency, 
#                  write_directory=None, early_stopping=True, early_key='val', print_progress=True):
#         self.evaluation_episodes = evaluation_episodes
#         self.model = model
#         self.eval_frequency = eval_frequency
#         self.write_directory = write_directory
#         self.early_stopping = early_stopping
#         self.early_key = early_key
#         self.best_accuracy = 0
#         self.print_progress = print_progress
#         self.curves = {key:[] for key in evaluation_episodes}

#     def check_early(self, accuracy):
#         if accuracy > self.best_accuracy:
#             self.best_accuracy = accuracy
#             # save best model to file
#             model_path = Path(self.write_directory, 'model.zip')
#             self.model.save(model_path)
        
#     def checkpoint_in(self, ckpt_dir):
#         self.best_accuracy = Utils.pickle_read(Path(ckpt_dir, 'curric__best_accuracy.p'))
#         self.difficulty_index = Utils.pickle_read(Path(ckpt_dir, 'curric__curves.p'))
    
#     def checkpoint_out(self, ckpt_dir):
#         Utils.pickle_write(Path(ckpt_dir, 'curric__best_accuracy.p'), self.best_accuracy)
#         Utils.pickle_write(Path(ckpt_dir, 'curric__curves.p'), self.curves)
#         Utils.json_write(Path(ckpt_dir, 'curric__curves.json'), self.curves)

#     def eval(self):
#         for key in self.evaluation_episodes:
#             episode = self.evaluation_episodes[key]
#             accuracy, episodes = Control.eval(episode, self.model)
#             self.curves[key].append(accuracy)
#         if self.early_stopping:
#             self.check_early(self.curves[self.early_key][-1])
#         if self.print_progress:
#             progress = f'episode: {self.n_episodes}  {self.early_key} accuracy: {self.best_accuracy:.2f}%'
#             job_name = Utils.get_global('job_name')
#             job_note = Utils.get_global('job_note')
#             if job_name is not None:
#                 Utils.update_progress(job_name, job_note + ' ' + progress)
#             print(progress)

#     def end(self, episode):
#         self.n_episodes = episode.get_episodes()
#         if self.n_episodes % self.eval_frequency == 0:
#             self.eval()

class HighLevelPolicy(Other):

    def __init__(self, agent, observer=None, model='gemma3:27b', silent=False, cuda_device=None, options={}, image_path=None, image_path2=None,
                 include_map=True, include_path_history=True, include_waypoints=True, include_state=True, chain_of_thought=False,
                 explore_map=True, progress_threshold=1, waypoint_threshold=4, n_points=8, generate_waypoints=True,
                 x_min=None, x_max=None, y_min=None, y_max=None, resolution=1, ground_truth_global_grid=None, show_occupancy_grid=True):
        self.agent = agent
        self.observer = observer
        self.model = model
        self.silent = silent
        if cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        self.options = options
        self.image_path = image_path
        self.image_path2 = image_path2
        self.include_state = include_state
        self.include_map = include_map
        self.include_path_history = include_path_history
        self.include_waypoints = include_waypoints
        self.chain_of_thought = chain_of_thought
        self.explore_map = explore_map
        self.show_occupancy_grid = show_occupancy_grid
        self.generate_waypoints = generate_waypoints
        self.progress_threshold = progress_threshold
        self.waypoint_threshold = waypoint_threshold
        self.n_points = n_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.x_n = int((self.x_max - self.x_min) / self.resolution)
        self.y_n = int((self.y_max - self.y_min) / self.resolution)
        self.ground_truth_global_grid = ground_truth_global_grid
        self.start_token = ''
        alpha = 1
        scale = 1
        scale2 = 4
        if self.include_map:
            self.start_token = ' (green cross)'
            self.start_plt = {'color': '#39FF14', 'marker': 'P', 's': 20*scale, 'alpha': alpha}
            self.start_plt2 = {'color': '#39FF14', 'marker': 'P', 's': 20*scale2, 'alpha': alpha}
        self.robot_token = ''
        if self.include_map:
            self.robot_token = ' (blue square)'
            self.robot_plt = {'color': '#00008B', 'marker': 's', 's': 10*scale, 'alpha': alpha}
            self.robot_plt2 = {'color': '#01F9C6', 'marker': 's', 's': 20*scale2, 'alpha': alpha}
        self.target_token = ''
        if self.include_map:
            self.target_token = ' (yellow star)'
            self.target_plt = {'color': '#FFFF00', 'marker': '*', 's': 10*scale, 'alpha': alpha}
            self.target_plt2 = {'color': '#FFFF00', 'marker': '*', 's': 20*scale2, 'alpha': alpha}
        self.history_token = ''
        if self.include_map and self.include_path_history:
            self.history_token = ' (purple lines and circles)'
            self.history_plt = {'color': '#800080', 'marker': 'o', 'linewidth': 1*scale, 'markersize': 2*scale, 'alpha': alpha} 
            self.history_plt2 = {'color': '#800080', 'marker': 'o', 'linewidth': 0.5*scale2, 'markersize': 2*scale2, 'alpha': alpha} 
        self.waypoints_token = ''
        if self.include_map:
            self.waypoints_token = ' (red triangles)'
            self.waypoint_plt = {'color': '#FF0000', 'marker': '^', 's': 10*scale, 'alpha': alpha}
            self.waypoint_plt2 = {'color': '#FF0000', 'marker': '^', 's': 10*scale2, 'alpha': alpha}
        self.pathways_token = ''
        if self.include_map:
            self.pathways_token = ' (gaps between black and white space)'
        self.safe_token = ''
        if self.include_map:
            self.safe_token = ' (black pixels)'
        
    # start of an episode
    def start(self, episode:Episode.Episode):
        self.stop_checking = False
        self.strategy_history = []
        if self.explore_map and self.include_map:
            self.world = World.World(self.x_max-self.x_min, self.y_max-self.y_min, self.x_min, self.y_min)
    
    # check if we iterate to next waypoint
    def step(self, episode:Episode.Episode):

        # update global map?
        if self.explore_map and self.include_map and self.show_occupancy_grid:
            # self.world.update_map_from_depth_with_pitch_roll(episode.get_depth_map(), episode.point)
            self.world.update(episode.get_depth_map(), episode.point)

        # make map for display purposes only
        self.generate_map(episode, False)

        # check waypoint progress
        if self.generate_waypoints:
            if episode.waypoint is not None and episode.point.distance(episode.waypoint) <= self.waypoint_threshold:
                episode.waypoint = None
            elif self.check_stuck(episode):
                self.generate_waypoint(episode)

    def generate_map(self, episode:Episode.Episode, for_input:bool):

        # make figure to paint visual representation on
        fig = Figure()
        canvas = FigureCanvasAgg(fig) # hidden figure drawn off screen since we save to file
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
        # draw global map grid (either fully or partially explored) to paint robot state info onto
        map_grid = np.zeros((self.x_max-self.x_min, self.y_max-self.y_min, 3), dtype=np.uint8)
        if self.show_occupancy_grid:
            if self.explore_map:
                world_grid = self.world.global_grid
                map_grid[world_grid==-1] = (128, 128, 128) # grey unknown space
                map_grid[world_grid==0] = (0, 0, 0) # black free space
                map_grid[world_grid==1] = (255, 255, 255) # white occupied space
            else:
                map_grid = self.ground_truth_global_grid.copy()
        map_grid = np.transpose(map_grid, (1, 0, 2))
        ax.imshow(map_grid, origin='lower')

        # get robot state information
        x_o, y_o = episode.start_point.x, episode.start_point.y
        x_r, y_r = episode.point.x, episode.point.y
        x_t, y_t = episode.target_point.x, episode.target_point.y

        ## draw robot state information onto map below

        # if for input to MLLM use indicated markers that are optimized for machine parsing of the image
        if for_input: 
            history_plt = self.history_plt
            waypoint_plt = self.waypoint_plt
            start_plt = self.start_plt
            robot_plt = self.robot_plt
            target_plt = self.target_plt
        # otherwise use indicated markers for display purposes only that are optimized for human eyes
        else:
            history_plt = self.history_plt2
            waypoint_plt = self.waypoint_plt2
            start_plt = self.start_plt2
            robot_plt = self.robot_plt2
            target_plt = self.target_plt2
        
        # draw path history
        if self.include_path_history:
            for point_idx in range(len(episode.path_history)-1):
                point1 = episode.path_history[point_idx]
                point2 = episode.path_history[point_idx+1]
                if point_idx == 0:
                    ax.plot([point1.x-self.x_min, point2.x-self.x_min], 
                            [point1.y-self.y_min, point2.y-self.y_min], **history_plt, zorder=1, label='Path History')
                else:
                    ax.plot([point1.x-self.x_min, point2.x-self.x_min], 
                            [point1.y-self.y_min, point2.y-self.y_min], **history_plt, zorder=1)

        # draw waypoint history
        if self.include_waypoints:
            for wp_idx, point in enumerate(episode.waypoint_history):
                if wp_idx == 0:
                    ax.scatter(point.x-self.x_min, point.y-self.y_min, **waypoint_plt, zorder=2, label='Waypoint History')
                else:
                    ax.scatter(point.x-self.x_min, point.y-self.y_min, **waypoint_plt, zorder=2)

        # draw single point information for start, robot, and target
        ax.scatter(x_o-self.x_min, y_o-self.y_min, **start_plt, zorder=3, label='Start')
        ax.scatter(x_r-self.x_min, y_r-self.y_min, **robot_plt, zorder=4, label='Drone')
        ax.scatter(x_t-self.x_min, y_t-self.y_min, **target_plt, zorder=5, label='Target')

        # add legend for display purposes only
        if not for_input:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)

        # if state is not included in linguistic commands then we need to draw an axis system on figure
        if self.include_state:
            ax.axis('off')
        else:
            # translate array index values to x,y positions
            interval = 40
            ax.set_xticks([i for i in range(0, self.x_n, interval)], 
                          [int(self.x_min + i*self.resolution) for i in range(0, self.x_n, interval)], rotation=90)
            ax.set_yticks([i for i in range(0, self.y_n, interval)], 
                          [int(self.y_min + i*self.resolution) for i in range(0, self.y_n, interval)])
            ax.set_xlabel('x-coordinate')
            ax.set_ylabel('y-coordinate')
                   
        # write to file
        if for_input:
            fig.savefig(self.image_path, bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig(self.image_path2, bbox_inches='tight', pad_inches=0)

    # determine if we generate a new waypoint based on if the agent is stuck
    def check_stuck(self, episode):

        # used to override if we check (this can avoid inefficient checks or infinite loops)
        if self.stop_checking:
            return False

        # check if we have enough points in window to check progress
        if len(episode.path_history) < self.n_points:
            return False

        # measure change in progress to goal over last n_points
        initial_distance_to_target = episode.path_history[-self.n_points].distance(episode.target_point)
        closest_distance_to_target = initial_distance_to_target
        for i in range(1, self.n_points+1):
            point = episode.path_history[-1*i]
            distance = point.distance(episode.target_point)
            closest_distance_to_target = min(closest_distance_to_target, distance)
        progress = initial_distance_to_target - closest_distance_to_target
        return progress < self.progress_threshold

    # query MLLM to generate intermediate waypoint to get unstuck and make progress towards target
    def generate_waypoint(self, episode):

        # add visual component to multimodal input? If yes then write image to file to read into MLLM
        if self.include_map:
           self.generate_map(episode, True)

        # include_state = True will include the robot state information in the linguistic command to the MLLM
        if self.include_state:
            
            # get robot single point state information
            x_o, y_o, theta_o = episode.start_point.x, episode.start_point.y, episode.start_point.yaw
            x_r, y_r, theta_r = episode.point.x, episode.point.y, episode.point.yaw
            x_t, y_t = episode.target_point.x, episode.target_point.y
            
            prompt_str = f'You are the supervisor for a robot tasked to navigate to a target position. '
            prompt_str += f'The robot can only see what is in front of it, but you have access to the full map and path history. '
            prompt_str += f'The robot can navigate on its own, but it has just gotten stuck on a complex set of obstacles and is asking for your help to generate an intermediate waypoint to get unstuck and make progress towards the target. '
            prompt_str += f'The robot can only make discrete movements forward of between 1 to 32 meters or rotate the direction it is facing by 90 degrees. '
            if self.include_map:
                prompt_str += f'See the given image for reference, which is an illustration of the current map and robot state. '
                if self.explore_map:
                    prompt_str += f'The provided image was created by updating an occupancy grid through ray tracing of the robots egocentric depth sensors, IMU, and GPS, thus is only partially observed. '
            prompt_str += f'The robot{self.robot_token} is at ({x_r}, {y_r}, {theta_r}) as measured in (meters, meters, degrees). '
            prompt_str += f'The target{self.target_token} is at ({x_t}, {y_t}). '
            prompt_str += f'The start{self.start_token} is at ({x_o}, {y_o}, {theta_o}). '
            if self.include_path_history and len(episode.path_history) > 0:
                prompt_str += f'The robot has traveled with a path history{self.history_token} of (x, y, theta) coordinates: ['
                for point in episode.path_history:
                    prompt_str += f'({point.x}, {point.y}, {point.yaw}) '
                prompt_str = prompt_str[:-1]
                prompt_str += ']. '
            if self.include_waypoints and len(episode.waypoint_history) > 0:
                prompt_str += f'Consider the previous attempted waypoints{self.waypoints_token}, which were generated to try and unstuck the robot to no avail, are at the following (x, y) coordinates: ['
                for waypoint in episode.waypoint_history:
                    prompt_str += f'({waypoint.x}, {waypoint.y}) '
                prompt_str = prompt_str[:-1]
                prompt_str += ']. '
            if self.chain_of_thought:
                prompt_str += f'Your previously output strategy reasoning for why previous waypoints were generated are:  ['
                for i in range(len(self.strategy_history)):
                    strategy = self.strategy_history[i]
                    waypoint = episode.waypoint_history[i]
                    prompt_str += f'{{previous strategy for waypoint at ({waypoint.x}, {waypoint.y}) = {strategy}}} '
                prompt_str = prompt_str[:-1]
                prompt_str += ']. '
            if self.include_map:
                #prompt_str += f'White pixels are obstacles. '
                prompt_str += f'Black space is unknown. '
                #if self.explore_map:
                 #   prompt_str += f'Gray pixels are unknown. '
                prompt_str += f'Only generate a waypoint at a safe{self.safe_token} (x, y) coordinate, and one that has not been attempted yet. '
                prompt_str += f'Identify any complex obstacles that may block the robot from reaching the target. '
            prompt_str += f'Specifically format your response as: [STRATEGY]: reason for generating the waypoint. [WAYPOINT]: (x, y). '
            prompt_str += f'Do not include any conversational filler. '

        # include_state = False will not include any state information in the linguistic command and instead rely on the visual map input to the MLLM for all state information
        else:
            prompt_str = f'You are the supervisor for a robot tasked to navigate to a target position. '
            prompt_str += f'The robot can only see what is in front of it, but you have access to the full map and path history. '
            prompt_str += f'The robot can navigate on its own, but it has just gotten stuck on a complex set of obstacles and is asking for your help to generate an intermediate waypoint to get unstuck and make progress towards the target. '
            prompt_str += f'The robot can only make discrete movements forward of between 1 to 32 meters or rotate the direction it is facing by 90 degrees. '
            if self.include_map:
                prompt_str += f'See the given image for reference, which is an illustration of the current map and robot state. '
                if self.explore_map:
                    prompt_str += f'The provided image was created by updating an occupancy grid through ray tracing of the robots egocentric depth sensors, IMU, and GPS, thus is only partially observed. '
            prompt_str += f'The robot is indicated by the {self.robot_token}). '
            prompt_str += f'The target is indicated by the {self.target_token}. '
            prompt_str += f'The start is indicated by the {self.start_token}. '
            prompt_str += f'The path history is indicated by the {self.history_token}. '
            prompt_str += f'Consider the previous attempted waypoints as indicated by {self.waypoints_token}, which were generated to try and unstuck the robot to no avail. '
            if self.include_map:
                #prompt_str += f'White pixels are obstacles. '
                prompt_str += f'Black space is unknown. '
                #if self.explore_map:
                 #   prompt_str += f'Gray pixels are unknown. '
                prompt_str += f'Only generate a waypoint at a safe{self.safe_token} (x, y) coordinate, and one that has not been attempted yet. '
            if self.include_map or self.include_path_history:
                prompt_str += f'Identify any complex obstacles that may block the robot from reaching the target. '
            prompt_str += f'Specifically format your response as: [STRATEGY]: reason for generating the waypoint. [WAYPOINT]: (x, y). '
            prompt_str += f'Do not include any conversational filler. '

        # query the MLLM with the generated prompt and get response
        response = self.chat(prompt_str, not self.silent, episode)
        episode.add_to_state('high_level_policy_response', response)

        # the output either contains [WAYPOINT] if one is generated or not
        waypoint_phrase = '[WAYPOINT]: '
        if waypoint_phrase not in response:
            return
        waypoint_idx = response.index(waypoint_phrase)

        # this indicates the reasoning into why the MLLM generated the given waypoint
        strategy_phrase = '[STRATEGY]:'
        if strategy_phrase in response:
            strategy_idx = response.index(strategy_phrase)
            if waypoint_idx > strategy_idx:
                strategy = response[strategy_idx:waypoint_idx].strip()
            else:
                strategy = response[strategy_idx:].strip()

        # parse waypoint coordinates from MLLM response and set waypoint if valid
        try:
            i = waypoint_idx+len(waypoint_phrase)+1
            substr = response[i:]
            i2 = substr.index(')')
            point_str = substr[:i2]
            x, y = point_str.split(', ')
            x = int(x)
            y = int(y)
            
            waypoint = DataStructure.Point(x, y, self.agent.fixed_z)
            current_goal = episode.waypoint if episode.waypoint is not None else episode.target_point
            if current_goal.x != waypoint.x or current_goal.y != waypoint.y:
                self.set_waypoint(episode, waypoint)
                if strategy_phrase in response:
                    self.strategy_history.append(strategy)
                else:
                    self.strategy_history.append('Unknown')
            else:
                self.stop_checking = True # otherwise this will now result in an infinite loop of generating the same waypoint
        except Exception as exception:
            pass

    # communicate with MLLM
    def chat(self, prompt, display_chat=True, episode=None):
        if self.include_map:
            message = {'role': 'user', 'content': prompt, 'images': [self.image_path]}
        else:
            message = {'role': 'user', 'content': prompt}
        response = ollama.chat(model=self.model, messages=[message], options=self.options)
        if display_chat:
            print(response.message.content)
        return response.message.content
        
    # set a new intermediate waypoint
    def set_waypoint(self, episode, waypoint):

        # tell episode we have a new goal to navigate towards
        episode.waypoint = waypoint

        # reset observations since we have a new goal, which will change the previously observed relative distances to goal in obervation FIFO queue
        if self.observer is not None:
            self.observer.start(episode)
