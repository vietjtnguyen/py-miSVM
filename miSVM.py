#!/usr/bin/python
import numpy as np
import sklearn.metrics
import sklearn.svm

class MultipleInstanceSVC():
	'''
	An implementation of miSVM [1] built on top of Scikit-Learn's SVM
	(`sklearn.svm.SVC`) to perform multiple-instance classification [2]. An
	instance-level classifier is trained from bag-level samples. This weakly-
	supervised formulation eases data labeling requirements.

	Although `sklearn.svm.SVC` supports multi-class classification, this
	implementation of miSVM is built for binary classification where the labels
	are -1 or +1.
	
	Parameters
	----------
	**kwargs :
	    All keyword arguments are passed to the SVC constructor. However,
	    the SVC keyword argument `probability` is forced to `True`.
	
	Attributes
	----------
	`num_of_fit_iterations_` : integer
	    Number of training iterations before convergence or max iterations.

	See also
	--------
	SVC
	    Implementation of Support Vector Machine classifier using `libsvm`.

	Notes
	----

	Not to be confused with MI-SVM which is also detailed in [1].

	[1] Support Vector Machines for Multiple-Instance Learning (Andrews et al 2002)
	[2] Solving the Multiple Instance Problem with Axis-Parallel Rectangles (Dietterich et al 1997)
	'''

	def __init__(self, **kwargs):
		self.svc_parameters = kwargs
		self.svc_parameters['probability'] = True
		self.svc = None
		self.num_of_fit_iterations_ = 0

	def fit(self, X, y, max_iterations=100, **kwargs):
		'''
		Fit the miSVM model according to the given bag-level training data.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
        	Bag-level training set, where n_bags is the number of bags. Each
        	item should be an array-like of samples of shape [n_samples,
        	n_features] where n_samples can vary between bags but n_features is
        	constant.

        y : array-like, shape = [n_bags]
            Target bag-level binary (+1/-1) class labels.

        max_iterations : integer, default = 100
            The maximum number of iterations to perform for instance-level
            label imputing.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
		Handling of all array-like parameters has not been tested yet. It is
		written to handle Python lists.
		'''
		bag_level_samples = X
		bag_level_targets = y

		# initialize containers to hold instance level samples for SVC
		instance_level_samples = []
		instance_level_targets = []
		bag_instance_index_lists = []

		# convert bags into instance level arrays
		instance_index = 0
		for bag_level_sample, bag_level_target in zip(bag_level_samples, bag_level_targets):
			bags_instance_level_samples = bag_level_sample

			instance_level_samples.extend(bags_instance_level_samples)
			instance_level_targets.extend([bag_level_target] * len(bags_instance_level_samples))

			# remember the instance level indices for bag level iteration over instance level arrays later
			bag_instance_indices = range(instance_index, instance_index + len(bags_instance_level_samples))
			bag_instance_index_lists.append(bag_instance_indices)

			instance_index += len(bags_instance_level_samples)

		instance_level_samples = np.array(instance_level_samples)
		instance_level_targets = np.array(instance_level_targets)

		# initialize the iteration targets
		iteration_instance_level_targets = np.copy(instance_level_targets)

		self.svc = None
		self.num_of_fit_iterations_ = 0

		while True:
			self.svc = sklearn.svm.SVC(**self.svc_parameters)
			self.svc.fit(instance_level_samples, iteration_instance_level_targets, **kwargs)
			instance_level_labels = self.svc.predict(instance_level_samples)
			instance_level_probabilities = self.svc.predict_proba(instance_level_samples)
		
			# perform bag-level clean up on predicted labels
			for bag_index, bag_instance_indices in enumerate(bag_instance_index_lists):
				bag_level_target = bag_level_targets[bag_index]
				if bag_level_target == -1:
					# if the bag is a negative bag then "clamp" all instance labels to negative
					instance_level_labels[bag_instance_indices] = -1
				else:
					# if all the positive bag instance_samples have negative labels then take the one
					# with the highest probability of being positive and set that to positive
					if np.sum(np.clip(instance_level_labels[bag_instance_indices], 0, 1)) == 0:
						max_probability_index = np.argmax(instance_level_probabilities[:, 1])
						instance_level_labels[max_probability_index] = 1
			
			if np.sum(np.abs(iteration_instance_level_targets - instance_level_labels)) == 0 or self.num_of_fit_iterations_ >= max_iterations:
				# kill the iterations if we've reached convergence or hit the max number of iterations
				break
			else:
				# otherwise use the bag-level cleaned labels as our training target for next iteration
				iteration_instance_level_targets = instance_level_labels

				# ensure the classifier gets cleaned up?
				del self.svc
			
			self.num_of_fit_iterations_ += 1

		return self

	def predict_in_bag(self, X):
		'''
		Returns the instance-level predicted label per bag on the given test
		bag-level data. 
		
		Performs instance-level prediction (see `predict_instances`) on each
		instance in each bag but returns the predicted labels per bag in an
		array.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        Returns
        -------
        y_pred : array, shape = [n_bags], items = ( array-like, shape = [n_samples] )
		'''
		assert(not self.svc == None)
		bag_level_samples = X
		bag_level_labels = []
		for bag_level_sample in bag_level_samples:
			bags_instance_level_samples = bag_level_sample
			bags_instance_level_labels = self.svc.predict(bags_instance_level_samples)
			bag_level_labels.append(bags_instance_level_labels)
		return bag_level_labels

	def predict(self, X):
		'''
		Perform classification on bag-level samples in X.
		
		An array of predicted +1 or -1 bag-level labels is returned.
		
		Parameters
		----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.
		
		Returns
		-------
		y_pred : array, shape = [n_bags]
		'''
		bag_level_labels = self.predict_in_bag(X)
		return [max(bags_instance_level_labels) for bags_instance_level_labels in bag_level_labels]

	def predict_instances(self, X):
		'''
		Perform classification on instance-level samples in X.
		
		An array of predicted +1 or -1 instance-level labels is returned.

		Just a pass-through to the trained SVC's `predict` method.
		
		Parameters
		----------
        X : array-like, shape = [n_samples, n_features]
            Instance-level test set.
		
		Returns
		-------
		y_pred : array, shape = [n_samples]
		'''
		return self.svc.predict(X)

	def score_in_bag(self, X, y):
		'''
		Returns the instance-level accuracy per bag on the given test bag-level data
		and per-bag-instance-level labels.

		Performs instance-level scoring (see `score_instances`) on each instance in
		each bag but returns the scores per bag in an array.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        y : array-like, shape = [n_bags], items = ( array-like, shape =
        [n_samples] )
            Bag-level labels for X.

        Returns
        -------
        z : array, shape = [n_bags]
		'''
		assert(not self.svc == None)
		bag_level_samples = X
		bag_level_targets = y
		bag_level_scores = []
		for bag_level_sample, bag_level_target in zip(bag_level_samples, bag_level_targets):
			bags_instance_level_samples = bag_level_sample
			bags_instance_level_targets = bag_level_target
			bags_instance_level_scores = self.svc.score(bags_instance_level_samples, bags_instance_level_targets)
			bag_level_scores.append(bags_instance_level_scores)
		return bag_level_scores

	def score(self, X, y):
		'''
		Returns the mean bag-level accuracy on the given test bag-level data and
		bag-level labels.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        y : array-like, shape = [n_bags]
            Bag-level labels for X.

        Returns
        -------
        z : float
		'''
		return sklearn.metrics.accuracy_score(self.predict(X), y)

	def score_instances(self, X, y):
		'''
		Returns the instance-level accuracy on the given test intance-level data
		and instance-level labels.

		Just a pass-through to the trained SVC's `score` method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Instance-level test set.

        y : array-like, shape = [n_samples]
            Instance-level labels for X.

        Returns
        -------
        z : float
		'''
		return self.svc.score(X)

