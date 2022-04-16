# -*- coding: utf-8 -*-
import os
os.environ["PBR_VERSION"]='3.1.1'
import tensorflow as tf
import numpy as np
import conv

class CNN(object):
	def __init__(
			self,
			weigth,
			gNet_hide_num,
			filter_size,
			num_classes,
			num_filters,
			is_imbalance_loss,
			learning_rate,
			l2_reg_lambda,
			a):
		self.l2_reg_lambda = l2_reg_lambda
		self.is_imbalance_loss = is_imbalance_loss
		self.a = a
		self.input_x = tf.placeholder(tf.float32, [None, filter_size*2+1, weigth], name = "input_x")
		self.input_y = tf.placeholder(tf.float32,[None, num_classes], name = "input_y")
		#self.gate_y = tf.placeholder(tf.float32,[None, heigth, num_classes], name = "gate_y")
		#gate_y_temp = []
		#for i in range(filter_size*2):
		#	gate_y_temp.append(tf.expand_dims(self.input_y, 0))
		#self.gate_y = tf.concat(gate_y_temp, 0)

		#print tf.shape(self.input_x)
	
		self.input_yn = tf.constant([[0,1]], dtype = tf.float32)*self.input_y
		self.input_yp = tf.constant([[1,0]], dtype = tf.float32)*self.input_y

		self.batch_size = tf.placeholder(tf.int32, shape=None, name="batch_size")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		#self.is_nogate = tf.placeholder(tf.bool, name="is_nogate")
		#self.is_train = tf.placeholder(tf.bool, name='is_train')

		self.l2_loss = tf.constant(0.0)

		with tf.name_scope("conv"):
			self.W_conv = tf.Variable(tf.truncated_normal([filter_size*2+1, weigth, num_filters], stddev = 0.001), name = "W_conv")

			self.b_conv = tf.Variable(tf.constant(0.0, shape=[num_filters]))

		with tf.name_scope("Full_net"):
			self.W_full = tf.get_variable(
					"W_full",  
					#shape = [conv_num*num_filters, num_classes],
					shape = [num_filters, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())
			self.b_full = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

		self.CNN_varlist = [self.W_conv, self.b_conv, self.W_full, self.b_full]
			
		'''
		with tf.name_scope("gateVarp"):
			self.Wg_p = tf.get_variable(
					"Wg_p",
					shape = [2*weigth, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())

			self.bg_p = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bg_p")

		with tf.name_scope("gateVarn"):
			self.Wg_n = tf.get_variable(
					"Wg_n",
					shape = [2*weigth, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())

			self.bg_n = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bg_n")
		'''

		with tf.name_scope("gateVarpu"):
			self.Wg_pu = tf.get_variable(
					"Wg_pu",
					shape = [weigth, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())

			self.bg_pu = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bg_pu")

		with tf.name_scope("gateVarnu"):
			self.Wg_nu = tf.get_variable(
					"Wg_nu",
					shape = [weigth, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())

			self.bg_nu = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bg_nu")



		#self.gate_varlistp = [self.Wg_p, self.bg_p]
		#self.gate_varlistn = [self.Wg_n, self.bg_n]

		#gate_y1 = tf.reshape(self.gate_y, [-1,num_classes])
		
		#self.gate_yn = tf.constant([[0,1]], dtype = tf.float32)*gate_y1
		#self.gate_yp = tf.constant([[1,0]], dtype = tf.float32)*gate_y1

		self.gate_varlistpu = [self.Wg_pu, self.bg_pu]
		self.gate_varlistnu = [self.Wg_nu, self.bg_nu]

		gate_xu = tf.reshape(self.input_x[:,filter_size,:], [-1,weigth])
		self.gate_yu = tf.reshape(self.input_y, [-1,num_classes])

		self.gate_ynu = tf.constant([[0,1]], dtype = tf.float32)*self.gate_yu
		#self.gate_ynu = self.gate_ynu[:,0] + self.gate_ynu[:,1]
		self.gate_ypu = tf.constant([[1,0]], dtype = tf.float32)*self.gate_yu
		#self.gate_ypu = self.gate_ypu[:,0] + self.gate_ypu[:,1]

		self.g_probpu = conv.gNetp_u(gate_xu, self.Wg_pu, self.bg_pu)
		self.g_probnu = conv.gNetn_u(gate_xu, self.Wg_nu, self.bg_nu)
		
		
		#self.input_1 = pad(self.input_x, weigth, filter_size[0])

		self.convp, self.convn, self.gatep, self.gaten = conv.cNet(
				self.input_x,
				self.W_conv,
				num_filters,
				weigth,
				filter_size,
				self.Wg_pu,
				self.bg_pu,
				self.Wg_nu,
				self.bg_nu,
				namep = "gatep",
				namen = "gaten")


		self.hp = tf.nn.tanh(tf.nn.bias_add(self.convp, self.b_conv),name="tanhp")
		self.hn = tf.nn.tanh(tf.nn.bias_add(self.convn, self.b_conv),name="tanhn")
		
		#self.h_flat = tf.reshape(self.h, [self.batch_size, -1], name="h_flat")
		self.h_p = tf.reshape(self.hp, [-1, num_filters], name = "h_p")
		self.h_n = tf.reshape(self.hn, [-1, num_filters], name = "h_n")
		#self.gNetp = tf.reshape(self.gNetp, [-1, num_classes])
		#self.gNetn = tf.reshape(self.gNetn, [-1, num_classes])

		h_pn = tf.stack([self.h_p,self.h_n],axis = 1)
		gate_yu_temp = tf.expand_dims(self.gate_yu,2)
		gate_yu_temp = tf.tile(gate_yu_temp,[1,1,num_filters])
		self.h_pn = h_pn*gate_yu_temp
		self.cv = h_pn[:,0]+h_pn[:,1]

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.cv, self.dropout_keep_prob, name="h_d")

		with tf.name_scope("CNN_output"):

			self.l2_loss += tf.nn.l2_loss(self.W_full)
			self.l2_loss += tf.nn.l2_loss(self.b_full)

			self.scores = tf.nn.xw_plus_b(self.h_drop, self.W_full, self.b_full, name="scores")
			self.logits = tf.nn.softmax(self.scores, name = "softmax")
			self.predictions = tf.argmax(self.logits, 1, name = "prediction")
		with tf.name_scope("loss"):
			if self.is_imbalance_loss:
				self.loss_n = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_yn)
				self.loss_p = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_yp)
				self.loss_mean_n = tf.reduce_sum(self.loss_n)/(tf.reduce_sum(self.input_yn)+1e-10)
				self.loss_mean_p = tf.reduce_sum(self.loss_p)/(tf.reduce_sum(self.input_yp)+1e-10)
				self.loss_cNet = tf.square(self.loss_mean_n) + tf.square(self.loss_mean_p)
			else:
				self.loss_n = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_yn)
				#self.loss_p = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_yp)
				self.loss_mean_n = tf.reduce_sum(self.loss_n)/(tf.reduce_sum(self.input_yn)+1e-10)
				#self.loss_mean_p = tf.reduce_sum(self.loss_p)/(tf.reduce_sum(self.input_yp)+1e-10)

				losses_cNet = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
				self.loss_cNet = tf.reduce_mean(losses_cNet)+self.l2_reg_lambda*self.l2_loss
			
			''
			self.loss_gatep_nu = -1*self.gate_ynu*tf.log(tf.clip_by_value(self.g_probpu, 1e-10, 1.0))
			self.loss_gatep_pu = -1*self.gate_ypu*tf.log(tf.clip_by_value(self.g_probpu, 1e-10, 1.0))
			self.loss_meangatep_nu = tf.reduce_sum(self.loss_gatep_nu)/(tf.reduce_sum(self.gate_ynu)+1e-10)
			self.loss_meangatep_pu = tf.reduce_sum(self.loss_gatep_pu)/(tf.reduce_sum(self.gate_ypu)+1e-10)
			self.loss_gNetpu = self.a*self.loss_meangatep_nu + self.loss_meangatep_pu
			#self.loss_gNetpu = self.loss_meangatep_nu + self.loss_meangatep_pu
			''	
			#loss_gNetpu = tf.nn.softmax_cross_entropy_with_logits(logits = self.g_probpu, labels = self.gate_yu)
			#self.loss_gNetpu = tf.reduce_mean(loss_gNetpu)

			''
			self.loss_gaten_nu = -1*self.gate_ynu*tf.log(tf.clip_by_value(self.g_probnu, 1e-10, 1.0))
			self.loss_gaten_pu = -1*self.gate_ypu*tf.log(tf.clip_by_value(self.g_probnu, 1e-10, 1.0))
			self.loss_meangaten_nu = tf.reduce_sum(self.loss_gaten_nu)/(tf.reduce_sum(self.gate_ynu)+1e-10)
			self.loss_meangaten_pu = tf.reduce_sum(self.loss_gaten_pu)/(tf.reduce_sum(self.gate_ypu)+1e-10)
			self.loss_gNetnu = 1000*self.loss_meangaten_nu + self.loss_meangaten_pu
			#self.loss_gNetnu = self.loss_meangaten_nu + self.loss_meangaten_pu
			''
			#loss_gNetnu = tf.nn.softmax_cross_entropy_with_logits(logits = self.g_probnu, labels = self.gate_yu)
			#self.loss_gNetnu = tf.reduce_mean(loss_gNetnu)

			#self.loss_gatep_n = -1*self.gate_yn*tf.log(tf.clip_by_value(self.gNetp, 1e-10, 1.0))
			#self.loss_gatep_p = -1*self.gate_yp*tf.log(tf.clip_by_value(self.gNetp, 1e-10, 1.0))
			#self.loss_meangatep_n = tf.reduce_sum(self.loss_gatep_n)/(tf.reduce_sum(self.gate_yn)+1e-10)
			#self.loss_meangatep_p = tf.reduce_sum(self.loss_gatep_p)/(tf.reduce_sum(self.gate_yp)+1e-10)
			#self.loss_gNetp = a*self.loss_meangatep_n + self.loss_meangatep_p
			
			#self.loss_gaten_n = -1*self.gate_yn*tf.log(tf.clip_by_value(self.gNetn, 1e-10, 1.0))
			#self.loss_gaten_p = -1*self.gate_yp*tf.log(tf.clip_by_value(self.gNetn, 1e-10, 1.0))
			#self.loss_meangaten_n = tf.reduce_sum(self.loss_gaten_n)/(tf.reduce_sum(self.gate_yn)+1e-10)
			#self.loss_meangaten_p = tf.reduce_sum(self.loss_gaten_p)/(tf.reduce_sum(self.gate_yp)+1e-10)
			#self.loss_gNetn = 1000*self.loss_meangaten_n + self.loss_meangaten_p
		

		with tf.name_scope("optimizer"):
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			
			#grads_and_vars_gatep = optimizer.compute_gradients(self.loss_gNetp, var_list = self.gate_varlistp)
			#grads_and_vars_gaten = optimizer.compute_gradients(self.loss_gNetn, var_list = self.gate_varlistn)
			grads_and_vars_gatepu = optimizer.compute_gradients(self.loss_gNetpu, var_list = self.gate_varlistpu)
			grads_and_vars_gatenu = optimizer.compute_gradients(self.loss_gNetnu, var_list = self.gate_varlistnu)
			grads_and_vars_cnn = optimizer.compute_gradients(self.loss_cNet, var_list = self.CNN_varlist)
			
			#self.train_op_gatep = optimizer.apply_gradients(grads_and_vars_gatep, global_step = self.global_step)
			#self.train_op_gaten = optimizer.apply_gradients(grads_and_vars_gaten, global_step = self.global_step)
			self.train_op_gatepu = optimizer.apply_gradients(grads_and_vars_gatepu, global_step = self.global_step)
			self.train_op_gatenu = optimizer.apply_gradients(grads_and_vars_gatenu, global_step = self.global_step)
			self.train_op_cnn = optimizer.apply_gradients(grads_and_vars_cnn, global_step = self.global_step)
			


		

