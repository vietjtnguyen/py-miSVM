#!/usr/bin/python
import scipy.io
import numpy as np
import sklearn.cross_validation
import miSVM

if __name__ == '__main__':

	# prepare data
	mat = scipy.io.loadmat('musk1_normalized_matlab.mat')
	instance_bag_ids = np.array(mat['bag_ids'])[0]
	instance_samples = np.array(mat['features'].todense())
	instance_targets = np.array(mat['labels'].todense())[0]
	num_of_instances = len(instance_samples)

	bag_samples = []
	bag_instance_targets = []
	bag_targets = []
	for bag_id in np.unique(instance_bag_ids):
		this_bag_instance_indices = filter(lambda instance_index: instance_bag_ids[instance_index] == bag_id, range(0, num_of_instances))
		this_bag_instances = [instance_samples[i] for i in this_bag_instance_indices]
		this_bag_instance_targets = [instance_targets[i] for i in this_bag_instance_indices]
		bag_samples.append(this_bag_instances)
		bag_instance_targets.append(this_bag_instance_targets)
		bag_targets.append(1 if np.max(this_bag_instance_targets) == 1 else -1)

	num_of_runs = 10
	num_of_folds = 10
	run_accuracies = np.zeros(num_of_runs)
	for run_index in range(num_of_runs):
		folds = sklearn.cross_validation.KFold(len(bag_samples), n_folds=num_of_folds, indices=True, shuffle=True)

		print('')
		print('Run #{:}'.format(run_index))
		
		fold_accuracies = np.zeros(len(folds))
		for fold_index, fold in enumerate(folds):
			training_indices, testing_indices = fold

			training_bag_samples = [bag_samples[i] for i in training_indices]
			training_bag_instance_targets = [bag_instance_targets[i] for i in training_indices]
			training_bag_targets = [bag_targets[i] for i in training_indices]

			testing_bag_samples = [bag_samples[i] for i in testing_indices]
			testing_bag_instance_targets = [bag_instance_targets[i] for i in testing_indices]
			testing_bag_targets = [bag_targets[i] for i in testing_indices]
			
			misvc = miSVM.MultipleInstanceSVC(kernel='rbf', gamma=0.015, C=1)
			misvc.fit(training_bag_samples, training_bag_targets)

			bag_accuracies = misvc.score_in_bag(testing_bag_samples, testing_bag_instance_targets)
			fold_accuracies[fold_index] = np.mean(bag_accuracies)
			print('')
			print('  Fold #{:}'.format(fold_index))
			print('    Bag accuracies:')
			for bag_accuracy in bag_accuracies:
				print('      {:.4%}'.format(bag_accuracy))
			print('    Fold accuracy: {:.4%}'.format(fold_accuracies[fold_index]))


		run_accuracies[run_index] = np.mean(fold_accuracies)
		print('')
		print('  Run accuracy: {:.4%}'.format(run_accuracies[run_index]))

	print('')
	print('Overall accuracy (standard deviation): {:.4%} ({:})'.format(np.mean(run_accuracies), np.std(run_accuracies)))

