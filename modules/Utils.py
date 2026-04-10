from pathlib import Path
import numpy as np
import torch as th
import platform
import datetime
import random
import pickle
import shutil
import psutil
import json
import time
import os

SET_DIRECTORIES_ON_IMPORT = True
SUB_DIRECTORY_NAMES = ['configs', 'data', 'models', 'modules', 'notebooks', 'results', 'scripts', 'policies']

# global PARAMS
global_parameters = {}
def set_globals(params):
	global_parameters.update(params)
def set_global(key, value):
	global_parameters[key] = value
def get_global(key):
	if key not in global_parameters:
		if key in SUB_DIRECTORY_NAMES:
			print(key, 'not set -- import utils module first')
		if key == 'job_note':
			return ''
		return None
	else:
		return global_parameters[key]

# auto set on import relative directories within repository
def set_directories():
	# get absolute path of this .py file
	file_path = Path(__file__).resolve()
	# get root repository directory from this file (assumes this is in repository/modules/utils.py)
	repository_directory = file_path.parent.parent
	print('repository_directory', repository_directory)
	# set globals for access throughout repository
	set_global('repository_directory', repository_directory)
	# set all sub directories
	for sub_dir in SUB_DIRECTORY_NAMES:
		set_global(f'{sub_dir}_directory', Path(repository_directory, sub_dir))
	# set local directory pulled out of dropbox and the github repository
	set_global('local_directory', Path('/home/tim/local'))
	
if SET_DIRECTORIES_ON_IMPORT:
	set_directories()
												
def check_ram(msg=''):
	print(f'{msg} {psutil.virtual_memory().used*1e-9:0.4f} gb')

def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory
# setup paths and some global params
def setup_output_dir(output_dir, overwrite_directory=False):
	# set working directory
	if overwrite_directory and os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir, exist_ok=True)
	# make temp folder if not exists
	#os.makedirs(f'{output_dir}temp/', exist_ok=True)
	# set operation system
	set_global('OS', platform.system().lower())
	# save working directory path to global_parameters to be visible by all 
	set_global('output_dir', output_dir) # relative to repo
	# absoulte path on local computer to repo
	set_global('absolute_path',  os.getcwd() + '/')

# used for argvs but can be used for whatevs
	# inputs string following dictionary format of 'key1:value1 key2:value2 ... keyN:valueN'
	# set_global will update global params dictionary variable with arguments dict
def isint(s):
	for c in s:
		if c not in ['-','0','1','2','3','4','5','6','7','8','9']:
			return False
	return True
def isfloat(s):
	for c in s:
		if c not in ['-','0','1','2','3','4','5','6','7','8','9','.','e']:
			return False
	return True
def args_to_str(args):
	s = ''
	for key in args:
		s += f'{key}:{args[key]} '
	return s
def parse_arguments(arguments, set_global_arguments=True):
	dictionary = {}
	for keyvalue in arguments:
		parts = keyvalue.split(':')
		key = parts[0]
		value = ':'.join(parts[1:])
		print(key, value)
		if value[0]=='{':
			value = parse_arguments(value[1:-1].split('__'), set_global_arguments=False)
		elif value[0]=='[':
			value = value[1:-1].split('__')
		elif value in ['True']:
			value = True
		elif value in ['False']:
			value = False
		elif isint(value):
			value = int(value)
		elif isfloat(value):
			value = float(value)
		dictionary[key] = value
		if set_global_arguments:
			set_global(key, value)
	return dictionary

# COMMUNICATE WITH USER
local_log = []
def add_to_log(msg):
	local_log.append(get_timestamp() + ': ' + str(msg))
	#print_local_log()
def print_local_log():
	file = open(get_global('output_dir') + 'log.txt', 'w')
	for item in local_log:
		file.write(item + "\n")
	file.close()
def speak(msg):
	add_to_log(msg)
	print(msg)
def prompt(msg):
	speak(msg)
	return get_user_input()
def get_user_input():
	return input()
def error(msg):
	add_to_log(msg)
	raise Exception('ERROR:', msg)
def warning(msg):
	speak('WARNING:', msg)

# **** common utility funcitons **** 
def set_random_seed(random_seed):
	random.seed(random_seed)
	np.random.seed(random_seed)
	th.manual_seed(random_seed)	
	if th.cuda.is_available():
		th.cuda.manual_seed_all(random_seed) 
	set_global('random_seed', random_seed)


# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class Stopwatch:
	def __init__(self):
		self.start()
	def start(self):
		self.start_time = time.time()
		self.last_time = self.start_time
		self.laps = []
	def lap(self):
		this_time = time.time()
		delta_time = this_time - self.last_time
		self.laps.append(delta_time)
		self.last_time = this_time
		return delta_time
	def stop(self):
		self.stop_time = time.time()
		self.delta_time = self.stop_time - self.start_time
		return self.delta_time
def pickle_read(path):
	return pickle.load(open(path, 'rb'))
def pickle_write(path, obj):
	pickle.dump(obj, open(path, 'wb'))
def json_read(path):
	return json.load(open(path, 'r'))
def json_write(path, dictionary):
	json.dump(dictionary, open(path, 'w'), indent=2)
def get_timestamp():
	secondsSinceEpoch = time.time()
	time_obj = time.localtime(secondsSinceEpoch)
	timestamp = '%d_%d_%d_%d_%d_%d' % (
		time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday,  
		time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec
	)
	return timestamp
def to_datetime(timestamp):
	format_string = '%Y_%m_%d_%H_%M_%S'
	datetime_object = datetime.datetime.strptime(timestamp, format_string)
	return datetime.datetime.timestamp(datetime_object)
	
def update_progress(name, progress):
	if name is None:
		return
	old_progress = get_global('progress')
	set_global('progress', progress)
	progress_dir = Path(get_global('local_directory'), 'progress')
	if not os.path.exists(progress_dir):
		os.makedirs(progress_dir)
	if old_progress is not None:
		old_path = Path(progress_dir, f'{name} {old_progress}.p')
		if os.path.exists(old_path):
			os.remove(old_path)
	new_path = Path(progress_dir, f'{name} {progress}.p')
	#print('write', new_path)
	pickle_write(new_path, '')

def unique_labels(axis):
	# Get handles and labels
	handles, labels = axis.get_legend_handles_labels()
	# Create a dictionary to store unique labels and their corresponding handles
	unique_labels = {}
	for handle, label in zip(handles, labels):
		if label not in unique_labels:
			unique_labels[label] = handle
	# Extract unique handles and labels in order
	unique_handles = list(unique_labels.values())
	unique_labels_list = list(unique_labels.keys())
	return unique_handles, unique_labels_list

    
# # handle fan speeds
# import pynvml
# def set_gpu_fan_full():
#     """Forces all GPU fans to 100%."""
#     try:
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
#         # Determine how many fans the card has
#         num_fans = pynvml.nvmlDeviceGetNumFans(handle)
        
#         for i in range(num_fans):
#             # Set each fan to 100%
#             pynvml.nvmlDeviceSetFanSpeed_v2(handle, i, 100)

#         time.sleep(2)
#         speed = pynvml.nvmlDeviceGetFanSpeed(handle)
#         print(f"Current reported speed: {speed}%")
            
#         print(f"Success: All {num_fans} fans set to 100%.")
#     except pynvml.NVMLError as err:
#         print(f"Error setting fan speed: {err}")
# def reset_gpu_fan_to_auto():
#     """Returns fan control to the hardware's default thermal curve."""
#     try:
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
#         # On modern NVML versions, setting 'Default' policy returns control to VBIOS
#         num_fans = pynvml.nvmlDeviceGetNumFans(handle)
#         for i in range(num_fans):
#             # 0 is the index for 'GpuFanControlMode_Auto' in many driver versions
#             pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, i)

#         time.sleep(2)
#         speed = pynvml.nvmlDeviceGetFanSpeed(handle)
#         print(f"Current reported speed: {speed}%")
        
#         pynvml.nvmlShutdown()
#         print("Success: Fan control returned to Auto.")
#     except pynvml.NVMLError as err:
#         print(f"Error resetting fan: {err}")
#         # Fallback: Just shut down NVML. The driver usually re-asserts 
#         # control when the handl
