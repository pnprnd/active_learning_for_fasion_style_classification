# -*- coding: utf-8 -*-
#http://necotech.org/archives/964

import cv2
import numpy as np

from matplotlib import pyplot as plt

import hyperparameters as hp
import bag_of_words as bow

def get_sift_descriptors(img):
	detector = cv2.xfeatures2d.SIFT_create(40)
	keypoints, descriptors = detector.detectAndCompute(img, None)
	#cv2.drawKeypoints(img, keypoints, img)
	#cv2.imshow("test", img)
	#cv2.waitKey(0)
	return keypoints, descriptors # (keynum * 128)

def get_bow_dictionary():
	words = bow.gen_words()
	return bow.make_clusters(words)

def get_bow_dictionary_abs(abs_paths):
	words = bow.gen_words_abs(abs_paths)
	return bow.make_clusters(words)

def bow_histogram(dictionary, img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	keypoints, descriptors = get_sift_descriptors(img)
	histogram = [0] * hp.vocab_size
	if len(keypoints) == 0: return histogram
	for d in descriptors:
		cluster = find_nearest_vocab(dictionary, d)
		histogram[cluster] = histogram[cluster] + 1
	return (histogram / np.linalg.norm(histogram)).tolist()	# normalize


def find_nearest_vocab(dictionary, descriptor):	# can do soft assignment
	diff = dictionary - descriptor
	ss = np.sum(diff**2, axis=1)
	i = np.argmax(ss)
	return i

def color_histogram(img, bin_size):
	chans = cv2.split(img)
	colors = ("b", "g", "r")
	feature = []

	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], None, [bin_size], [0, 256])
		feature.extend(hist)
	return [x[0] for x in np.array(feature)]

def get_features(img_ids, type):
	features = []

	if type == hp.FeatureType.COLOR_HIST:
		for img_id in img_ids:
			img = cv2.imread(hp.cur_dir + "labeled_images/" + img_id + ".jpg")
			feature = color_histogram(img, 256)
			features.append(feature)

	elif type == hp.FeatureType.BOW:
		dictionary = get_bow_dictionary()
		for img_id in img_ids:
			img = cv2.imread(hp.cur_dir + "labeled_images/" + img_id + ".jpg")
			feature = bow_histogram(dictionary, img)
			features.append(feature)

	elif type == hp.FeatureType.COL_BOW:
		dictionary = get_bow_dictionary()
		for img_id in img_ids:
			img = cv2.imread(hp.cur_dir + "labeled_images/" + img_id + ".jpg")
			f1 = color_histogram(img, 50)
			f2 = bow_histogram(dictionary, img)
			features.append(f1 + f2)

	return features

def get_features_abs(dictionary, abs_paths, type):
	features = []

	if type == hp.FeatureType.COLOR_HIST:
		for path in abs_paths:
			img = cv2.imread(path)
			feature = color_histogram(img, 256)
			features.append(feature)

	elif type == hp.FeatureType.BOW:
		for path in abs_paths:
			img = cv2.imread(path)
			feature = bow_histogram(dictionary, img)
			features.append(feature)

	elif type == hp.FeatureType.COL_BOW:
		for path in abs_paths:
			img = cv2.imread(path)
			f1 = color_histogram(img, 50)
			f2 = bow_histogram(dictionary, img)
			features.append(f1 + f2)

	return features