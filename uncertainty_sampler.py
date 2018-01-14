# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

import hyperparameters as hp

def maximum_entropy(probs, select_num):
	data_num = probs.shape[0]
	entropies = []
	for i in range(data_num):
		s = 0
		for prob in probs[i,:]:
			s = s - prob * np.log2(prob if prob > 0 else 0.00001)
		entropies.append(s)
	return np.argsort(entropies)[::-1][:select_num]

def smallest_margin(probs, select_num):
	data_num = probs.shape[0]
	diff = []
	for i in range(data_num):
		sorted_probs = sorted(probs[i,:])[::-1]
		diff.append(sorted_probs[0] - sorted_probs[1])
	return np.argsort(diff)[:select_num]

def least_confidence(probs, select_num):
	data_num = probs.shape[0]
	lc = []
	for i in range(data_num):
		lc.append(min(1 - probs[i,:]))
	return np.argsort(lc)[::-1][:select_num]

def select(classifier, features, select_num, type):
	if features == []: return []
	probs = classifier.predict_proba(features)
	if type == hp.SamplerType.MAXIMUM_ENTROPY:
		return maximum_entropy(probs, select_num)
	elif type == hp.SamplerType.SMALLEST_MARGIN:
		return smallest_margin(probs, select_num)
	elif type == hp.SamplerType.LEAST_CONFIDENCE:
		return least_confidence(probs, select_num)
