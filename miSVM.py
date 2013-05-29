#!/usr/bin/python
import numpy as np
import sklearn.metrics
import sklearn.svm

def create_bags_from_instance_ids(instance_samples, instance_targets, instance_bag_ids):
	'''
	This utility function will create well-formed bags using bag IDs associated
	with every instance.

	Parameters
	----------
	instance_samples : array-like, shape = [n_samples, n_features]

	instance_targets : array-like, shape = [n_samples]

	instance_bag_ids : array-like, shape = [n_samples], items = ( integer )

	Returns
	-------
	bag_samples : array, shape = [n_bags], items = ( array-like, shape = [bag_n_samples, n_features] )

	bag_targets : array, shape = [n_bags], items = ( array-like, shape = [bag_n_samples, n_features] )

	bag_instance_targets : array, shape = [n_bags], items = ( array-like, shape = [bag_n_samples] )

	'''
	num_of_instances = len(instance_samples)
	bag_samples = []
	bag_targets = []
	bag_instance_targets = []
	for bag_id in np.unique(instance_bag_ids):
		this_bag_instance_indices = filter(lambda instance_index: instance_bag_ids[instance_index] == bag_id, range(0, num_of_instances))
		this_bag_instances = [instance_samples[i] for i in this_bag_instance_indices]
		this_bag_instance_targets = [instance_targets[i] for i in this_bag_instance_indices]
		bag_samples.append(this_bag_instances)
		bag_instance_targets.append(this_bag_instance_targets)
		bag_targets.append(1 if np.max(this_bag_instance_targets) == 1 else -1)
	return bag_samples, bag_targets, bag_instance_targets

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

	def fit(self, X, y, max_iterations=100, K=None, per_iter_func=None, **kwargs):
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

        max_iterations : integer, optional (default=100)
            The maximum number of iterations to perform for instance-level
            label imputing.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

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
		kernel = K

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

		if not kernel == None:
			subset_kernel = kernel[instance_level_samples, :][:, instance_level_samples]

		while True:
			if not per_iter_func == None:
				per_iter_func(self.num_of_fit_iterations_, iteration_instance_level_targets)

			self.svc = sklearn.svm.SVC(**self.svc_parameters)
			if kernel == None:
				self.svc.fit(instance_level_samples, iteration_instance_level_targets, **kwargs)
				instance_level_labels = self.svc.predict(instance_level_samples)
				instance_level_probabilities = self.svc.predict_proba(instance_level_samples)
			else:
				self.precomputed_kernel_training_indices_ = instance_level_samples
				self.svc.fit(subset_kernel, iteration_instance_level_targets, **kwargs)
				instance_level_labels = self.svc.predict(subset_kernel)
				instance_level_probabilities = self.svc.predict_proba(subset_kernel)
		
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

	def predict_in_bag(self, X, K=None):
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

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

        Returns
        -------
        y_pred : array, shape = [n_bags], items = ( array-like, shape = [n_samples] )
		'''
		assert(not self.svc == None)
		bag_level_samples = X
		bag_level_labels = []
		for bag_level_sample in bag_level_samples:
			bags_instance_level_samples = bag_level_sample
			if K == None:
				bags_instance_level_labels = self.svc.predict(bags_instance_level_samples)
			else:
				subset_kernel = K[bags_instance_level_samples, :][:, self.precomputed_kernel_training_indices_]
				bags_instance_level_labels = self.svc.predict(subset_kernel)
			bag_level_labels.append(bags_instance_level_labels)
		return bag_level_labels

	def predict(self, X, K=None):
		'''
		Perform classification on bag-level samples in X.
		
		An array of predicted +1 or -1 bag-level labels is returned.
		
		Parameters
		----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.
		
		Returns
		-------
		y_pred : array, shape = [n_bags]
		'''
		assert(not self.svc == None)
		bag_instance_labels = self.predict_in_bags(X, K=K)
		return [max(bags_instance_level_labels) for bags_instance_level_labels in bag_instance_labels]

	def predict_instances(self, X, K=None):
		'''
		Perform classification on instance-level samples in X.
		
		An array of predicted +1 or -1 instance-level labels is returned.

		Just a pass-through to the trained SVC's `predict` method unless a K is
		specified.
		
		Parameters
		----------
        X : array-like, shape = [n_samples, n_features]
            Instance-level test set.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.
		
		Returns
		-------
		y_pred : array, shape = [n_samples]
		'''
		assert(not self.svc == None)
		if K == None:
			return self.svc.predict(X)
		else:
			subset_kernel = K[X, :][:, self.precomputed_kernel_training_indices_]
			return self.svc.predict(subset_kernel)

	def predict_proba_in_bag(self, X, K=None):
		'''
		Returns the instance-level predicted class probabilities per bag on the
		given test bag-level data. 
		
		Performs instance-level prediction (see `predict_proba_instances`) on
		each instance in each bag but returns the predicted class probabilities
		per bag in an array.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

        Returns
        -------
        y_pred : array, shape = [n_bags], items = ( array-like, shape = [n_samples, 2] )
		'''
		assert(not self.svc == None)
		bag_level_samples = X
		bag_level_probas = []
		for bag_level_sample in bag_level_samples:
			bags_instance_level_samples = bag_level_sample
			if K == None:
				bags_instance_level_probas = self.svc.predict_proba(bags_instance_level_samples)
			else:
				subset_kernel = K[bags_instance_level_samples, :][:, self.precomputed_kernel_training_indices_]
				bags_instance_level_probas = self.svc.predict_proba(subset_kernel)
			bag_level_probas.append(bags_instance_level_probas)
		return bag_level_probas

	def predict_proba_instances(self, X, K=None):
		'''
		Just a pass-through to the trained SVC's `predict_proba` method unless a K is
		specified.
		
		Parameters
		----------
        X : array-like, shape = [n_samples, n_features]
            Instance-level test set.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.
		
		Returns
		-------
		y_pred : array, shape = [n_samples]
		'''
		assert(not self.svc == None)
		if K == None:
			return self.svc.predict_proba(X)
		else:
			subset_kernel = K[X, :][:, self.precomputed_kernel_training_indices_]
			return self.svc.predict_proba(subset_kernel)

	def score_in_bag(self, X, y, K=None):
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

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

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
			bags_instance_level_labels = self.predict_instances(bags_instance_level_samples, K=K)
			bags_instance_level_scores = sklearn.metrics.accuracy_score(bags_instance_level_labels, bags_instance_level_targets)
			bag_level_scores.append(bags_instance_level_scores)
		return bag_level_scores

	def score(self, X, y, K=None):
		'''
		Returns the mean bag-level accuracy on the given test bag-level data and
		bag-level labels.

        Parameters
        ----------
        X : array, shape = [n_bags], items = ( array-like, shape = [n_samples, n_features] )
            Bag-level test set.

        y : array-like, shape = [n_bags]
            Bag-level labels for X.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

        Returns
        -------
        z : float
		'''
		assert(not self.svc == None)
		return sklearn.metrics.accuracy_score(self.predict(X, K=K), y)

	def score_instances(self, X, y, K=None):
		'''
		Returns the instance-level accuracy on the given test intance-level data
		and instance-level labels.

		Just a pass-through to the trained SVC's `score` method unless a K is
		specified.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Instance-level test set.

        y : array-like, shape = [n_samples]
            Instance-level labels for X.

        K : array-like, sparse matrix, shape = [total_n_samples, total_n_samples], options, (default=None)
            Distance kernel (Gram matrix) for precomputed kernels. If a precomputed
            kernel is used then the bag instances should be indices to the
            instances appropriate row in the distance kernel. This means each bag
            will simply be an array-like of integers meaning the effective
            n_features will be 1.

        Returns
        -------
        z : float
		'''
		assert(not self.svc == None)
		return sklearn.metrics.accuracy_score(self.predict_instances(X, K=K), y)

