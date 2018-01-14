from enum import Enum

class ClassifierType(Enum):
	RANDOM_FOREST = 0

class FeatureType(Enum):
	COLOR_HIST = 0
	BOW = 1
	COL_BOW = 2

class SamplerType(Enum):
	MAXIMUM_ENTROPY = 0
	SMALLEST_MARGIN = 1
	LEAST_CONFIDENCE = 2

cur_dir = "./"
max_steps = 200
vocab_size = 200
tree_depth = 6
tree_num = 200
sampling_ratio = 0.05