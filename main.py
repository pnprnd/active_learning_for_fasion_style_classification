# -*- coding: utf-8 -*-

import numpy as np
import random
import os.path

import feature_extractor as fe
import classifier as cf
import uncertainty_sampler as us
import hyperparameters as hp

image_list = hp.cur_dir + "item_styling_label.txt"

def load_labeled_images():
	images, labels = [], []
	with open(image_list, 'r') as f:
		for i in f:
			line_list = i.strip().split(' ')
			item_id = line_list[0]
			label = line_list[1]
			images.append(item_id)
			labels.append(label)
	return images, labels

def get_precision(classifier, fts, lbs):
	if len(lbs) == 0: return 0
	predicts = classifier.predict(fts)
	precision = 0
	for p, l in zip(predicts, lbs):
		precision = precision + (1 if p == l else 0)
	#return "precision: %.2lf (%d, %d)" % (float(precision)/len(lbs)*100, precision, len(lbs))
	return [float(precision)/len(lbs)*100, precision, len(lbs)]

if __name__ == '__main__':

	##################################################
	#	LOAD DATA DIRECTORIES
	##################################################
	images, labels = load_labeled_images()
	all_img_num = len(images)
	print ">> data loaded"

	##################################################
	#	SPLIT TRAIN-TEST DATA
	##################################################
	shuffled_ids = range(all_img_num)
	random.seed(5)
	random.shuffle(shuffled_ids)
	test_ratio = 0.2
	testset = 4
	train_ids = shuffled_ids[:int(all_img_num*test_ratio)*testset]+shuffled_ids[int(all_img_num*test_ratio)*(testset+1):]
	test_ids = shuffled_ids[int(all_img_num*test_ratio)*testset:int(all_img_num*test_ratio)*(testset+1)]
	train_img_num = len(train_ids)
	print ">> train-test split (%d, %d)" % (len(train_ids), len(test_ids))

	##################################################
	#	EXTRACT FEATURES
	##################################################
	fdir = hp.cur_dir + "tmp/features_colbow.csv"
	if os.path.isfile(fdir):
		print ">> use existing features"
		features = np.genfromtxt(fdir, delimiter=',')
	else:
		print ">> extract features"
		features = fe.get_features(images, hp.FeatureType.COL_BOW) # imgNum*featureLen
		np.savetxt(fdir, features, delimiter=',')
		print ">> features extracted"

	##################################################
	#	FIT A CLASSIFIER USING RANDOM DATA
	##################################################
	all_data = train_ids
	sampled_num = int(train_img_num*hp.sampling_ratio)
	sampled_ids = all_data[:sampled_num]
	lbs = [labels[x] for x in sampled_ids]
	fts = [features[x] for x in sampled_ids]
	classifier = cf.get_classifier(fts, lbs, hp.ClassifierType.RANDOM_FOREST)
	print ">> fit classifier with %d labeled samples" % sampled_num

	train_features = [features[x] for x in train_ids]
	train_labels = [labels[x] for x in train_ids]
	test_features = [features[x] for x in test_ids]
	test_labels = [labels[x] for x in test_ids]
	print "train", get_precision(classifier, train_features, train_labels)
	print "test", get_precision(classifier, test_features, test_labels)

	##################################################
	#	ACTIVE LEARNING / RANDOM SAMPLING
	##################################################
	useAL = True 	# use Active Learning or Random Sampling (when comparing AL with random sampling, do AL first)
	comparePrecision = False 	# True when # of random sampling data = # of AL data
	step = 0
	samples_per_step = sampled_num
	precisions = []
	if comparePrecision:
		if useAL: data_num = []
		else: data_num = np.genfromtxt(hp.cur_dir + "tmp/datanum.csv", delimiter=',')
	print
	print "START ACTIVE LEARNING"
	print
	while step < hp.max_steps and len(all_data) > 0:
		print "*** STEP", step, "***"
		if useAL:
			cur_features = [features[x] for x in all_data]
			ids = us.select(classifier, cur_features, samples_per_step, hp.SamplerType.SMALLEST_MARGIN)
			selected_ids = [all_data[x] for x in ids]
			all_data = [x for x in all_data if x not in selected_ids] # remove selected from all_data
		else:
			if comparePrecision and step > len(data_num)-1: break
			if comparePrecision: samples_per_step = int(data_num[step])
			selected_ids = all_data[:samples_per_step]
			if not comparePrecision: all_data = all_data[samples_per_step:]
		if useAL and comparePrecision: data_num.append(len(train_ids)-len(all_data))
		print ">> %d data selected" % (len(train_ids)-len(all_data))

		lbs = lbs + [labels[x] for x in selected_ids]
		fts = fts + [features[x] for x in selected_ids]

		classifier = cf.get_classifier(fts, lbs, hp.ClassifierType.RANDOM_FOREST)
		print ">> classifier fit"

		train_pre = get_precision(classifier, train_features, train_labels)
		test_pre = get_precision(classifier, test_features, test_labels)
		print "train", train_pre
		print "test", test_pre
		precisions.append(train_pre + test_pre)
		print
		step = step + 1
	np.savetxt(hp.cur_dir + "tmp/precisions.csv", precisions, delimiter=',')
	if useAL and comparePrecision: np.savetxt(hp.cur_dir + "tmp/datanum.csv", data_num, delimiter=',')
