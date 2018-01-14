# -*- coding: utf-8 -*-

import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import hyperparameters as hp

def get_classifier(features, labels, type):
	if type == hp.ClassifierType.RANDOM_FOREST:
		return random_forest(features, labels)

def random_forest(features, labels):
	classifier = RandomForestClassifier(max_depth=hp.tree_depth, n_estimators=hp.tree_num, random_state=0)
	classifier.fit(features, labels)
	return classifier