# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
os.environ["PBR_VERSION"]='3.1.1'
import sys
import data_helpers
from CNN import CNN
from AutoEncoderBenign import AutoEncoderBenign
from AutoEncoderAttack import AutoEncoderAttack
from collections import OrderedDict
from correlationNet import correlationNet

def training(basepath, filter_size, num_dim):

	FILTER_SIZES = int(filter_size)
	fs = '1'
	a = 0.001

	USERNAME = os.listdir(basepath)
	LEARNING_RATE = 0.01
	DROPOUT_KEEP_PROB = 0.6
	L2_REG_LAMBDA = 0.0
	NUM_CLASSES = 2
	BATCH_SIZE = 50
	EPOCHS = 6

	IS_IMBALANCE_LOSS = True
	#NUM_DIM = 1195
	NUM_DIM = int(num_dim)
	#FILTER_SIZES = 8
	NUM_FILTERS = 5
	GNET_HIDE_NUM = 300
	POSITION_RATE = 0.8
	NEGATION_RATE = 0.7
	IS_RATE = True


	CHECKPOINT_EVERY = EPOCHS * BATCH_SIZE
	NUM_CHECKPOINTS = 5

	if not os.path.exists('./to'):
		os.makedirs('./to/model')
		os.makedirs('./to/modelB')

	MODEL_SAVE_PATH = "./to/model"
	MODEL_NAME = "model.ckpt"
	#result = open("./loss.csv","w")

	testfile = open('./testfile.csv','wb')
	testlabel = open('./testlabel.csv','wb')
	testfile.truncate()
	testlabel.truncate()

	data_list = []
	date_list = []
	label_list = []
	lc_list = []
	label_truth = []
	unknown_num = 0
	negation_num = 0
	#f_ra = open('./representation_a_train.txt','w')
	#f_rb = open('./representation_b_train.txt','w')
	for i in USERNAME:
	
		POSITION_PATH = basepath+"/"+i+"/position/"
		NEGATION_PATH = basepath+"/"+i+"/negation/"
		LABEL_POSITION_PATH = basepath+"/"+i+"/labelposition/"
		LABEL_NEGATION_PATH = basepath+"/"+i+"/labelnegation/"

		if IS_RATE:
			data_path_list, _, label_path_list, _ = data_helpers.train_or_test_rate(
					position_path = POSITION_PATH,
					negation_path = NEGATION_PATH,
					label_position_path = LABEL_POSITION_PATH,
					label_negation_path = LABEL_NEGATION_PATH,
					position_rate = POSITION_RATE,
					negation_rate = NEGATION_RATE)
		else:
			data_path_list, label_path_list = data_helpers.train_or_test(
					position_path = POSITION_PATH,
					negation_path = NEGATION_PATH,
					label_position_path = LABEL_POSITION_PATH,
					label_negation_path = LABEL_NEGATION_PATH)
	
		data_user_list = []
		label_user_list = []
		lc_user_list = []
		label_user_truth = []
		for index, pathi in  enumerate(data_path_list):
			data_i, label_i, _, l_i = data_helpers.data_load_1(pathi, label_path_list[index], NUM_CLASSES,i)
			#NUM_DIM = len(data_i[0])
			x_i, lc_i = data_helpers.generating_x(np.array(data_i),np.array(label_i),FILTER_SIZES)
			data_user_list = data_user_list+x_i
			label_user_list = label_user_list+label_i
			label_user_truth = label_user_truth+l_i
			lc_user_list = lc_user_list+lc_i

		data_list = data_list+data_user_list
		label_list = label_list+label_user_list
		label_truth = label_truth+label_user_truth
		lc_list = lc_list+lc_user_list

	x = np.array(data_list)
	y = np.array(label_list)
	lt = np.array(label_truth)
	lc = np.array(lc_list)
	x_pair,y_pair = data_helpers.generating_pair_1(x,lt,FILTER_SIZES)


	lossdir = OrderedDict()
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
		
			cnn = CNN(
					weigth = NUM_DIM,
					gNet_hide_num = GNET_HIDE_NUM,
					filter_size = FILTER_SIZES,
					num_classes = NUM_CLASSES,
					num_filters = NUM_FILTERS,
					is_imbalance_loss = IS_IMBALANCE_LOSS,
					learning_rate = LEARNING_RATE,
					l2_reg_lambda = L2_REG_LAMBDA,
					a = a)

			autoencoderbenign = AutoEncoderBenign(
					num_filters = NUM_FILTERS,
					learning_rate = LEARNING_RATE)

			autoencoderattack = AutoEncoderAttack(
					num_filters = NUM_FILTERS,
					learning_rate = LEARNING_RATE)
		
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			threshold_b = []
			threshold_a = []
			def train_step(x_batch, y_batch, epoch):

				x_batch_n = []
				for i_batch in x_batch:
					x_batch_n_m = []
					for i in i_batch:
						weigth = [0]*NUM_DIM
						if i != 0:
							weigth[i-1] = 1
						x_batch_n_m.append(weigth)
					x_batch_n.append(x_batch_n_m)
					

				feed_dict_gatepu = {
						cnn.input_x: x_batch_n,
						cnn.input_y: y_batch,
						cnn.batch_size: len(x_batch_n),
						cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
				_, loss_gatepu, g_probpu = sess.run([cnn.train_op_gatepu, cnn.loss_gNetpu, cnn.g_probpu], feed_dict_gatepu)

				feed_dict_gatenu = {
						cnn.input_x: x_batch_n,
						cnn.input_y: y_batch,
						cnn.batch_size: len(x_batch_n),
						cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
				_, loss_gatenu, g_probnu = sess.run([cnn.train_op_gatenu, cnn.loss_gNetnu, cnn.g_probnu], feed_dict_gatenu)

				feed_dict_cnn = {
						cnn.input_x: x_batch_n,
						cnn.input_y: y_batch,
						cnn.batch_size:len(x_batch_n),
						cnn.dropout_keep_prob: DROPOUT_KEEP_PROB}
				_, step, loss_cnn, loss_n, representation, y = sess.run([cnn.train_op_cnn, cnn.global_step, cnn.loss_cNet, cnn.loss_mean_n, cnn.h_pn, cnn.gate_yu], feed_dict_cnn)
			

				rep_b = []
				rep_a = []
				for index, rep in enumerate(representation):
					if y[index,0] == 1:
						rep_b.append(rep[0].tolist())
					else:
						rep_a.append(rep[1].tolist())
				#print representation
			
				if rep_b:
					feed_dict_autoencoderbenign = {
							autoencoderbenign.representation_b: rep_b}
					_, autoencoderloss_b = sess.run([autoencoderbenign.train_op_autoencoder_b, autoencoderbenign.autoencoderloss_b], feed_dict_autoencoderbenign)

				else:
					autoencoderloss_b = 'None'
	
				if rep_a:
					feed_dict_autoencoderattack = {
							autoencoderattack.representation_a: rep_a}
					_, autoencoderloss_a = sess.run([autoencoderattack.train_op_autoencoder_a, autoencoderattack.autoencoderloss_a], feed_dict_autoencoderattack)
				else:
					autoencoderloss_a = 'None'
			
				epoch = str(epoch)
				if lossdir.has_key(epoch):
					lossdir[epoch].append(loss_n)
				else:
					lossdir[epoch] = [loss_n]
				#print epoch, loss_n
		
			batches = data_helpers.batch_iter(zip(x, y),BATCH_SIZE, EPOCHS)
		
			for batch in batches:
				#print epoch[index]:q
				x_batch, y_batch = zip(*batch[1])
				x_batch = np.array(list(x_batch))
				#x_batch = x_batch.reshape([len(x_batch), LENGTH, -1])
			
				train_step(x_batch,y_batch,batch[0])
			
				current_step = tf.train.global_step(sess, cnn.global_step)
				if current_step % CHECKPOINT_EVERY == 0:
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = current_step)

	BATCH_SIZE = 5
	EPOCHS = 20
	CHECKPOINT_EVERY = EPOCHS * BATCH_SIZE
	MODELB_SAVE_PATH = "./to/modelB"
	MODELB_NAME = "modelb.ckpt"
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
		
			corNet = correlationNet(
					weigth = NUM_DIM,
					num_classes = NUM_CLASSES,
					learning_rate = LEARNING_RATE)
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			def cor_train_step(x_batch, y_batch, epoch):
				
				x_batch_n = []
				for i in x_batch:
					weigth = [0]*NUM_DIM
					weigth[i[0]-1] = 1
					if i[0] == i[1]:
						weigth[i[1]-1] = 2
					else:
						 weigth[i[1]-1] = 1
					x_batch_n.append(weigth)

				feed_dict_cp = {
						corNet.input_x: x_batch_n,
						corNet.input_y: y_batch}
				_, loss_cp = sess.run([corNet.train_op_p, corNet.loss_p], feed_dict_cp)
			
				#print loss_cn, loss_cp
				#print loss_cp

			batches = data_helpers.batch_iter(zip(x_pair,y_pair),BATCH_SIZE, EPOCHS)

			for batch in batches:
				x_batch, y_batch = zip(*batch[1])
				x_batch = np.array(list(x_batch))
				cor_train_step(x_batch,y_batch,batch[0])
			
				current_step = tf.train.global_step(sess, corNet.global_step)
				if current_step % CHECKPOINT_EVERY == 0:
					saver.save(sess, os.path.join(MODELB_SAVE_PATH, MODELB_NAME),global_step = current_step)

