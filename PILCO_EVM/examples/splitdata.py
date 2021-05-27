import random
import numpy as np
import torch

def load_data(filename1, filename2, split_loc, time_series_len):
	data = np.load(filename1, allow_pickle=True)

	random.shuffle(data)

	train_data = data[:split_loc]
	test_data = data[split_loc:]

	train_data_split = []
	array_by_time_series_len = []
	count = 0
	for run in train_data:
		if len(run)%time_series_len != 0:
			current = run[:-(len(run)%time_series_len)]
		else:
			current = run
		for time_step in current:
			count+=1
			# array_by_time_series_len.append(time_step)
			array_by_time_series_len = array_by_time_series_len+time_step
			if count == time_series_len:
				count = 0
				array_by_time_series_len.insert(0,0)
				train_data_split.append(array_by_time_series_len)
				array_by_time_series_len = []


	data = np.load(filename2, allow_pickle=True)
	random.shuffle(data)
	train_data = data[:split_loc]
	test_data = data[split_loc:]

	test_data_split = []
	array_by_time_series_len = []
	count = 0
	for run in test_data:
		if len(run)%time_series_len != 0:
			current = run[:-(len(run)%time_series_len)]
		else:
			current = run
		for time_step in current:
			count+=1
			# array_by_time_series_len.append(time_step)
			array_by_time_series_len = array_by_time_series_len+time_step
			if count == time_series_len:
				count = 0
				array_by_time_series_len.insert(0,0)
				test_data_split.append(array_by_time_series_len)
				array_by_time_series_len = []


	second_train_data_split = []
	array_by_time_series_len = []
	count = 0
	for run in train_data:
		if len(run)%time_series_len != 0:
			current = run[:-(len(run)%time_series_len)]
		else:
			current = run
		for time_step in current:
			count+=1
			# array_by_time_series_len.append(time_step)
			array_by_time_series_len = array_by_time_series_len+time_step
			if count == time_series_len:
				count = 0
				array_by_time_series_len.insert(0,1)
				second_train_data_split.append(array_by_time_series_len)
				array_by_time_series_len = []


	second_test_data_split = []
	array_by_time_series_len = []
	count = 0
	for run in test_data:
		if len(run)%time_series_len != 0:
			current = run[:-(len(run)%time_series_len)]
		else:
			current = run
		for time_step in current:
			count+=1
			# array_by_time_series_len.append(time_step)
			array_by_time_series_len = array_by_time_series_len+time_step
			if count == time_series_len:
				count = 0
				array_by_time_series_len.insert(0,1)
				second_test_data_split.append(array_by_time_series_len)
				array_by_time_series_len = []

	combined_train_data_split = second_train_data_split + train_data_split
	combined_test_data_split = second_test_data_split + test_data_split
	random.shuffle(combined_train_data_split)
	random.shuffle(combined_test_data_split)
	# from IPython import embed; embed()

	return torch.from_numpy(np.asarray(combined_train_data_split)), torch.from_numpy(np.asarray(combined_test_data_split))

training_data, test_data = load_data('100_reg.npy', '100_pole_02.npy', split_loc=80, time_series_len=5)
