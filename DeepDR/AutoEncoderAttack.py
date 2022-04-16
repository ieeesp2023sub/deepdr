import os
os.environ["PBR_VERSION"]='3.1.1'
import tensorflow as tf
import numpy as np

class AutoEncoderAttack(object):
	def __init__(
			self,
			num_filters,
			learning_rate):

		self.representation_a = tf.placeholder(tf.float32,[None,num_filters], name = 'representation_a')
		
		with tf.name_scope("autoencoder_wa"):
			self.weights_a = {
					"encoder_w1":tf.Variable(tf.random_normal([num_filters,3])),
					"encoder_w2":tf.Variable(tf.random_normal([3,2])),
					"decoder_w1":tf.Variable(tf.random_normal([2,3])),
					"decoder_w2":tf.Variable(tf.random_normal([3,num_filters]))}
			self.biases_a = {
					"encoder_b1":tf.Variable(tf.random_normal([3])),
					"encoder_b2":tf.Variable(tf.random_normal([2])),
					"decoder_b1":tf.Variable(tf.random_normal([3])),
					"decoder_b2":tf.Variable(tf.random_normal([num_filters]))}
			var_list = self.weights_a.values() + self.biases_a.values()

		with tf.name_scope("encoder_a"):
			self.encoder_layer1 = tf.nn.sigmoid(
					tf.matmul(self.representation_a,self.weights_a["encoder_w1"])+self.biases_a["encoder_b1"])

			self.encoder_layer2 = tf.nn.sigmoid(
					tf.matmul(self.encoder_layer1,self.weights_a["encoder_w2"])+self.biases_a["encoder_b2"])

		with tf.name_scope("decoder_a"):
			self.decoder_layer1 = tf.nn.sigmoid(
					tf.matmul(self.encoder_layer2,self.weights_a["decoder_w1"])+self.biases_a["decoder_b1"])

			self.decoder_layer2 = tf.nn.sigmoid(
					tf.matmul(self.decoder_layer1,self.weights_a["decoder_w2"])+self.biases_a["decoder_b2"])

		with tf.name_scope("autoencoderloss_a"):
			self.autoencoderloss_a = tf.reduce_mean(tf.pow(self.representation_a - self.decoder_layer2,2), name='autoencoderloss_a')

		with tf.name_scope("autoencoder_optimizer_a"):
			autoencoder_optimizer_a = tf.train.AdamOptimizer(learning_rate)
			grads_and_vars_autoencoder_a = autoencoder_optimizer_a.compute_gradients(self.autoencoderloss_a, var_list = var_list)
			self.train_op_autoencoder_a = autoencoder_optimizer_a.apply_gradients(grads_and_vars_autoencoder_a)





