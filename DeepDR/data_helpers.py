# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import itertools
from PIL import Image

def ImageToMatrix(filename):
	im = Image.open(filename)
	width,height = im.size
	#im = im.covert('L')
	data = im.getdata()
	data = np.matrix(data,dtype='float')/255.0
	new_data = np.reshape(data,(height,width))
	return new_data

def train_or_test_negation(negation_path,label_negation_path):
	negation_array = np.array(os.listdir(negation_path))
	negation_shuffle_indices = np.random.permutation(np.arange(len(negation_array)))
	negation_array_shuffled = negation_array[negation_shuffle_indices]
	negation_train = negation_array_shuffled
	negation_train_list = [negation_path+i for i in negation_train]
	label_negation_train_list = [label_negation_path+i for i in negation_train]
	return negation_train_list, label_negation_train_list

def train_or_test(
		position_path,
		negation_path,
		label_position_path,
		label_negation_path):

	position_array = np.array(os.listdir(position_path))
	negation_array = np.array(os.listdir(negation_path))
	position_shuffle_indices = np.random.permutation(np.arange(len(position_array)))
	negation_shuffle_indices = np.random.permutation(np.arange(len(negation_array)))
	position_array_shuffled = position_array[position_shuffle_indices]
	negation_array_shuffled = negation_array[negation_shuffle_indices]

	position_train = position_array_shuffled

	negation_train = negation_array_shuffled


	position_train_list = [position_path+i for i in position_train]
	negation_train_list = [negation_path+i for i in negation_train]

	label_position_train_list = [label_position_path+i for i in position_train]
	label_negation_train_list = [label_negation_path+i for i in negation_train]

	return (
			position_train_list+negation_train_list,
			label_position_train_list+label_negation_train_list)

def train_or_test_position_rate(
		position_path,
		negation_path,
		label_position_path,
		label_negation_path,
		position_rate):

	position_array = np.array(os.listdir(position_path))
	negation_array = np.array(os.listdir(negation_path))
	position_shuffle_indices = np.random.permutation(np.arange(len(position_array)))
	negation_shuffle_indices = np.random.permutation(np.arange(len(negation_array)))
	position_array_shuffled = position_array[position_shuffle_indices]
	negation_array_shuffled = negation_array[negation_shuffle_indices]

	position_train = position_array_shuffled[int(len(position_array)*position_rate):]

	negation_train = negation_array_shuffled

	position_train_list = [position_path+i for i in position_train]
	negation_train_list = [negation_path+i for i in negation_train]
	
	label_position_train_list = [label_position_path+i for i in position_train]
	label_negation_train_list = [label_negation_path+i for i in negation_train]
	
	return (
			position_train_list+negation_train_list,
			label_position_train_list+label_negation_train_list)

def train_or_test_rate(
		position_path,
		negation_path,
		label_position_path,
		label_negation_path,
		position_rate,
		negation_rate):

	position_array = np.array(os.listdir(position_path))
	negation_array = np.array(os.listdir(negation_path))
	position_shuffle_indices = np.random.permutation(np.arange(len(position_array)))
	negation_shuffle_indices = np.random.permutation(np.arange(len(negation_array)))
	position_array_shuffled = position_array[position_shuffle_indices]
	negation_array_shuffled = negation_array[negation_shuffle_indices]

	position_train = position_array_shuffled[0:int(len(position_array)*position_rate)]
	position_test = position_array_shuffled[int(len(position_array)*position_rate):]
	negation_train = negation_array_shuffled[0:int(len(negation_array)*negation_rate)]
	negation_test = negation_array_shuffled[int(len(negation_array)*negation_rate):]

	position_train_list = [position_path+i for i in position_train]
	negation_train_list = [negation_path+i for i in negation_train]
	position_test_list = [position_path+i for i in position_test]
	negation_test_list = [negation_path+i for i in negation_test]

	label_position_train_list = [label_position_path+i for i in position_train]
	label_negation_train_list = [label_negation_path+i for i in negation_train]
	label_position_test_list = [label_position_path+i for i in position_test]
	label_negation_test_list = [label_negation_path+i for i in negation_test]

	test_list = position_test_list + negation_test_list
	test_label_list = label_position_test_list + label_negation_test_list
	#test_list = position_test_list
	#test_label_list = label_position_test_list
	
	'''
	test_label_list1 = []
	for i in test_label_list:
		if 'labelposition_semi' in i:
			a = i.replace('labelposition_semi','labelposition')
		elif 'labelnegation_semi' in i:
			a = i.replace('labelnegation_semi','labelnegation')
		else:
			a = i
		test_label_list1.append(a)
	'''

	testfile = open('./testfile.csv','a')
	testlabel = open('./testlabel.csv','a')

	testfile.write(','.join(test_list))
	testfile.write(',')

	testlabel.write(','.join(test_label_list))
	testlabel.write(',')

	return (
			position_train_list+negation_train_list,
			position_test_list+negation_test_list,
			label_position_train_list+label_negation_train_list,
			label_position_test_list+label_negation_test_list)

def data_load(data_path, label_path, num_class):
	data = []
	label = []
	date = []
	data_text = open(data_path)
	data_line = data_text.readline().rstrip()
	while data_line != '':
		part = data_line.split(',')
		data.append(int(part[-1]))
		date.append(part[0])
		data_line = data_text.readline().rstrip()

	label_text = open(label_path)
	label_line = label_text.readline()
	label_list = label_line.split(',')
	label_list = [int(i.rstrip()) for i in label_list]
	for i in label_list:
		t = [0]*num_class
		t[i] = 1
		label.append(t)
		
	return data, label, date, label_list

def data_load_1(data_path, label_path, num_class,username):
	data = []
	label = []
	date = []
	data_text = open(data_path)
	data_line = data_text.readline().rstrip()
	while data_line != '':
		part = data_line.split(',')
		data.append(int(part[-1]))
		date.append(part[0]+'_'+username)
		data_line = data_text.readline().rstrip()
	label_text = open(label_path)
	label_line = label_text.readline()
	label_list = label_line.split(',')
	label_list = [int(i.rstrip()) for i in label_list]
	for i in label_list:
		t = [0]*num_class
		k = i
		if i>1:
			k = 1
		t[k] = 1
		label.append(t)
	return data, label, date, label_list

def unknown_label(label_list, unknown_rate):
	negation_local = {}
	label_index = 0
	for index, label in enumerate(label_list):
		if label == [0,1]:
			negation_local[label_index] = index
			label_index += 1
	negation_num = label_index+1
	unknown_num = int(round(negation_num*unknown_rate))
	unknowns = random.sample(range(0,negation_num-1),unknown_num)
	for i in unknowns:
		label_list[negation_local[i]] = [1,0]
	return label_list, (unknown_num,negation_num)

def unknown_label2(data_list, label_list, unknown_rate):
	negation_local = {}
	negation_num = 0
	for index, label in enumerate(label_list):
		if label == [0,1]:
			negation_num += 1
			data = list(data_list[index][len(data_list[index])/2])
			elelist = []
			for i,ele in enumerate(data):
				if ele == 1:
					elelist.append(str(i))
			event_sty = ':'.join(elelist)
			if negation_local.has_key(event_sty):
				negation_local[event_sty].append(index)
			else:
				negation_local[event_sty] = [index]
	eventtypes = negation_local.keys()
	type_num = len(eventtypes)
	unknowns = random.sample(range(0,type_num-1),int(round(type_num*unknown_rate)))
	unknown_num = 0
	for i in unknowns:
		for l in negation_local[eventtypes[i]]:
			label_list[l] = [1,0]
			unknown_num += 1

	return label_list, (unknown_num,negation_num)

def generating_x(data,label,window_size):
	x = []
	l = []
	pad_zeros = np.full((window_size),0)
	pad_n = np.full((window_size,2),-1)
	pad_data = np.r_[pad_zeros,data,pad_zeros]
	pad_label = np.r_[pad_n,label,pad_n]
	start=window_size
	end=len(pad_data)-window_size-1
	i = start
	while i <= end:
		x.append(pad_data[i-window_size:i+window_size+1])
		l.append(pad_label[i-window_size:i+window_size+1])
		i+=1
	return x,l

def generating_x_l(data,label,date,window_size):
	x = []
	l = []
	d = []
	pad_zeros = np.full((window_size),0)
	pad_data = np.r_[pad_zeros,data,pad_zeros]
	pad_n1 = np.array([-1]*window_size)
	pad_label = np.r_[pad_n1,label,pad_n1]
	pad_dn1 = np.array([-1]*window_size)
	pad_date = np.r_[pad_dn1,date,pad_dn1]

	start=window_size
	end=len(pad_data)-window_size-1
	i = start
	while i <= end:
		x.append(pad_data[i-window_size:i+window_size+1])
		l.append(pad_label[i-window_size:i+window_size+1])
		d.append(pad_date[i-window_size:i+window_size+1])
		i+=1
	return x, l, d

def generating_pair(data,label,filter_size):
	x = []
	l = []
	for index, data_i in enumerate(data):
		x_main = data_i[filter_size,:]
		l_main = label[index][filter_size,:]
		for k in range(filter_size*2+1):
			if k != filter_size:
				x_line = data_i[k,:]
				l_line = label[index][k,:]
				if l_line[1] == -1:continue
				x.append(x_main+x_line)
				if l_main[1] == 1 and l_line[1] == 1:
					l.append([0,1])
				else:
					l.append([1,0])
	return x,l

def generating_pair_1(data,label,filter_size):
	x = []
	for index,l_i in enumerate(label):
		if l_i != 0:
			if data[index][filter_size] != 0:
				x.append((data[index][filter_size],l_i))

	pairs = list(itertools.combinations(x,2))
	pairs = list(set(pairs))
	x = []
	l = []
	for pair in pairs:
		l1 = pair[0][1]
		l2 = pair[1][1]
		x.append((pair[0][0],pair[1][0]))
		if l1 == l2:
			l.append([0,1])
		else:
			l.append([1,0])
	return x,l


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size)+1
	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1)*batch_size, data_size)
			yield epoch, shuffled_data[start_index:end_index]


	


