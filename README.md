# Fasion Style Classifier
This project shows the implementation of an active learning classifier for classifying fashion items into four styles according to "Styling Map" [1]. Active learning is a semi-supervised learning algorithm for interactive training. It is advantageous when we have a huge amount of unlabeled data and manually labeling them is expensive. By selecting data close to decision boundaries as ones to be lebeled next, we can fit a good classifier despite of the small amount of labeled training data. Uncertainty Sampling (maximum entropy, smallest margin, and least confidence) is implemented for selecting data. Bags of words (using SIFT descriptors and k-means) and color histograms are used as image features. We use the random forest algorithm for style classification.

# Reference
[1] https://stylist-kyokai.jp/stylingmap/map.html
