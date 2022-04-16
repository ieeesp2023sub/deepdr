import os
os.environ["PBR_VERSION"]='3.1.1'
import tensorflow as tf
import numpy as np

class AutoEncoderBenign(object):
	def __init__(
			self,
			num_filters,
			learning_rate):

		self.representation_b = tf.placeholder(tf.float32,[None,num_filters], name = 'representation_b')
		
		with tf.name_scope("autoencoder_wb"):
			self.weights_b = {
					"encoder_w1":tf.Variable(tf.random_normal([num_filters,3])),
					"encoder_w2":tf.Variable(tf.random_normal([3,2])),
					"decoder_w1":tf.Variable(tf.random_normal([2,3])),
					"decoder_w2":tf.Variable(tf.random_normal([3,num_filters]))}
			self.biases_b = {
					"encoder_b1":tf.Variable(tf.random_normal([3])),
					"encoder_b2":tf.Variable(tf.random_normal([2])),
					"decoder_b1":tf.Variable(tf.random_normal([3])),
					"decoder_b2":tf.Variable(tf.random_normal([num_filters]))}
			var_list = self.weights_b.values() + self.biases_b.values()

		with tf.name_scope("encoder_b"):
			self.encoder_layer1 = tf.nn.sigmoid(
					tf.matmul(self.representation_b,self.weights_b["encoder_w1"])+self.biases_b["encoder_b1"])

			self.encoder_layer2 = tf.nn.sigmoid(
					tf.matmul(self.encoder_layer1,self.weights_b["encoder_w2"])+self.biases_b["encoder_b2"])

		with tf.name_scope("decoder_b"):
			self.decoder_layer1 = tf.nn.sigmoid(
					tf.matmul(self.encoder_layer2,self.weights_b["decoder_w1"])+self.biases_b["decoder_b1"])

			self.decoder_layer2 = tf.nn.sigmoid(
					tf.matmul(self.decoder_layer1,self.weights_b["decoder_w2"])+self.biases_b["decoder_b2"])

		with tf.name_scope("autoencoderloss_b"):
			self.autoencoderloss_b = tf.reduce_mean(tf.pow(self.representation_b - self.decoder_layer2,2), name='autoencoderloss_b')

		with tf.name_scope("autoencoder_optimizer_b"):
			autoencoder_optimizer_b = tf.train.AdamOptimizer(learning_rate)
			grads_and_vars_autoencoder_b = autoencoder_optimizer_b.compute_gradients(self.autoencoderloss_b, var_list = var_list)
			self.train_op_autoencoder_b = autoencoder_optimizer_b.apply_gradients(grads_and_vars_autoencoder_b)





