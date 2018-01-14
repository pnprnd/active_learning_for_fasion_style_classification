# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import os, os.path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import hyperparameters as hp

def load_images():
	images = []
	path = hp.cur_dir + "labeled_images"
	for f in os.listdir(path):
		ext = os.path.splitext(f)[1]
		if ext.lower() != ".jpg": continue
		img = cv2.imread(path + "/" + f)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	# grayscale
		img = cv2.equalizeHist(img)
		images.append(img)
	return images

def gen_words():
	images = load_images()
	words = []
	for img in images:
		keypoints, descriptors = get_sift_descriptors(img)
		if len(keypoints) == 0: continue	# keypoint detection fails with white items
		words.extend(descriptors)
	return words

def gen_words_abs(abs_paths):
	words = []
	for path in abs_paths:
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	# grayscale
		img = cv2.equalizeHist(img)
		keypoints, descriptors = get_sift_descriptors(img)
		if len(keypoints) == 0: continue	# keypoint detection fails with white items
		words.extend(descriptors)
	return words

def make_clusters(data):
	clusterer = KMeans(n_clusters=hp.vocab_size)
	pred = clusterer.fit_predict(data)
	centers = clusterer.cluster_centers_
	return centers

def get_sift_descriptors(img):
	detector = cv2.xfeatures2d.SIFT_create()
	keypoints, descriptors = detector.detectAndCompute(img, None)
	return keypoints, descriptors # (keynum * 128)

