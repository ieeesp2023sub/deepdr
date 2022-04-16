import os
os.environ["PBR_VERSION"]='3.1.1'
import tensorflow as tf
import numpy as np

class correlationNet(object):
	def __init__(
			self,
			weigth,
			num_classes,
			learning_rate):

		self.input_x = tf.placeholder(tf.float32, [None, weigth], name = "cor_x")
		self.input_y = tf.placeholder(tf.float32,[None, num_classes], name = "cor_y")


		with tf.name_scope("cor_parameter"):
			Wp1 = tf.get_variable(
					"Wp1",
					shape = [weigth, 100],
					initializer = tf.contrib.layers.xavier_initializer())
	
			bp1 = tf.Variable(tf.constant(0.1, shape=[100]), name="bp1")

			Wp2 = tf.get_variable(
					"Wp2",
					shape = [100,num_classes],
					initializer = tf.contrib.layers.xavier_initializer())
			bp2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bp2")

			Wn = tf.get_variable(
					"Wn",
					shape = [weigth, num_classes],
					initializer = tf.contrib.layers.xavier_initializer())

			bn = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bn")

		self.varlistp = [Wp1, bp1, Wp2, bp2]
		#self.varlistn = [Wn, bn]

		self.input_y = tf.reshape(self.input_y, [-1,num_classes])
		self.input_yn = tf.constant([[0,1]], dtype = tf.float32)*self.input_y
		self.input_yp = tf.constant([[1,0]], dtype = tf.float32)*self.input_y
		
		layer1 = tf.nn.relu(tf.matmul(self.input_x,Wp1)+bp1)
		#cp = tf.matmul(self.input_x, Wp1) + bp1
		#cn = tf.nn.softmax(tf.matmul(self.input_x, Wn) + bn)
		cp_softmax = tf.nn.softmax(tf.matmul(layer1,Wp2)+bp2)
		
		#self.cp = tf.reshape(cp, [-1, num_classes])
		self.cp = tf.reshape(cp_softmax,[-1,num_classes],'score_cp')
		#self.cn = tf.reshape(cn, [-1, num_classes],'score_cn')
		
		self.loss_p_n = -1*self.input_yn*tf.log(tf.clip_by_value(self.cp, 1e-10, 1.0))
		self.loss_p_p = -1*self.input_yp*tf.log(tf.clip_by_value(self.cp, 1e-10, 1.0))
		self.loss_mean_p_n = tf.reduce_sum(self.loss_p_n)/(tf.reduce_sum(self.input_yn)+1e-10)
		self.loss_mean_p_p = tf.reduce_sum(self.loss_p_p)/(tf.reduce_sum(self.input_yp)+1e-10)
		self.loss_p = self.loss_mean_p_n + self.loss_mean_p_p

		#loss_p = tf.nn.softmax_cross_entropy_with_logits(logits = self.cp, labels = self.input_y)
		#self.loss_p = tf.reduce_mean(loss_p)

		#self.loss_n_n = -1*self.input_yn*tf.log(tf.clip_by_value(self.cn, 1e-10, 1.0))
		#self.loss_n_p = -1*self.input_yp*tf.log(tf.clip_by_value(self.cn, 1e-10, 1.0))
		#self.loss_mean_n_n = tf.reduce_sum(self.loss_n_n)/(tf.reduce_sum(self.input_yn)+1e-10)
		#self.loss_mean_n_p = tf.reduce_sum(self.loss_n_p)/(tf.reduce_sum(self.input_yp)+1e-10)
		#self.loss_n = 1000*self.loss_mean_n_n + self.loss_mean_n_p

		with tf.name_scope("cor_optimizer"):
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			grads_and_vars_p = optimizer.compute_gradients(self.loss_p, var_list = self.varlistp)
			#grads_and_vars_n = optimizer.compute_gradients(self.loss_n, var_list = self.varlistn)
			self.train_op_p = optimizer.apply_gradients(grads_and_vars_p, global_step = self.global_step)
			#self.train_op_n = optimizer.apply_gradients(grads_and_vars_n, global_step = self.global_step)



