import os
os.environ["PBR_VERSION"]='3.1.1'
import tensorflow as tf
import numpy as np
'''
def gNetp(x_line, Wg_p, bg_p):
	#g_hp = tf.nn.relu(tf.matmul(x, self.Wg_p) + self.bg_p)
	#x = tf.concat([x_line, x_main], 1)
	g_probp = tf.nn.softmax(tf.matmul(x_line, Wg_p) + bg_p)
	return g_probp

def gNetn(x_line, Wg_n, bg_n):
	#g_hn = tf.nn.relu(tf.matmul(x, Wg_n) + bg_n)
	#x = tf.concat([x_line, x_main], 1)
	g_probn = tf.nn.softmax(tf.matmul(x_line, Wg_n) + bg_n)
	return g_probn
'''
def gNetp_u(x_main, Wg_pu, bg_pu):
	g_probp_u = tf.nn.softmax(tf.matmul(x_main, Wg_pu) + bg_pu)
	return g_probp_u

def gNetn_u(x_main, Wg_nu, bg_nu):
	g_probn_u = tf.nn.softmax(tf.matmul(x_main, Wg_nu) + bg_nu)
	return g_probn_u

def cNet(x_conv, W, num_filters, weigth, filter_size, Wg_pu, bg_pu, Wg_nu, bg_nu, namep, namen):

	W_sum_p = 0
	W_sum_n = 0
	gate_kp = []
	gate_kn = []
	#gNetp_temp1 = []
	#gNetn_temp1 = []
	gate_filterp = []
	gate_filtern = []
	x_main = x_conv[:,filter_size,:]
	gate_linepu = gNetp_u(x_main, Wg_pu, bg_pu)
	gate_linepu0 = tf.reshape(gate_linepu[:,0],[1,-1,1])
	gate_linepu1 = tf.reshape(gate_linepu[:,1],[1,-1,1])
	gate_linepu0 = 1.0/2*(tf.sign(gate_linepu0-0.5)+1)
	gate_linepu1 = 1.0/2*(tf.sign(gate_linepu1-0.5)+1)
	gate_linenu = gNetn_u(x_main, Wg_nu, bg_nu)
	gate_linenu0 = tf.reshape(gate_linenu[:,0],[1,-1,1])
	gate_linenu1 = tf.reshape(gate_linenu[:,1],[1,-1,1])
	gate_linenu0 = 1.0/2*(tf.sign(gate_linenu0-0.5)+1)
	gate_linenu1 = 1.0/2*(tf.sign(gate_linenu1-0.5)+1)
	gatepu = gate_linepu0*gate_linenu0
	gatenu = gate_linepu1*gate_linenu1
	gate_nu = gatenu[0,:,:] #1 is anomaly
	gate_pu = gatepu[0,:,:] #1 is normal
	gate_u = 1 - (gate_nu+gate_pu) #1 is ambiguous event

	for k in range(filter_size*2+1):
		if k != filter_size:
			x_line = x_conv[:,k,:]
			gate_linep = gNetp_u(x_line, Wg_pu, bg_pu)
			gate_linep0 = tf.reshape(gate_linep[:,0],[1,-1,1])
			gate_linep1 = tf.reshape(gate_linep[:,1],[1,-1,1])
			gate_linep0 = 1.0/2*(tf.sign(gate_linep0-0.5)+1)
			gate_linep1 = 1.0/2*(tf.sign(gate_linep1-0.5)+1)
			gate_linen = gNetn_u(x_line, Wg_nu, bg_nu)
			gate_linen0 = tf.reshape(gate_linen[:,0],[1,-1,1])
			gate_linen1 = tf.reshape(gate_linen[:,1],[1,-1,1])
			gate_linen0 = 1.0/2*(tf.sign(gate_linen0-0.5)+1)
			gate_linen1 = 1.0/2*(tf.sign(gate_linen1-0.5)+1)
			gate_filterp.append(gate_linep0*gate_linen0) #normal
			gate_filtern.append(gate_linep1*gate_linen1) #anomaly
			#gNetp_temp1.append(tf.expand_dims(gate_linep, 0))
			#gNetn_temp1.append(tf.expand_dims(gate_linen, 0))
			
		#m = 1-tf.reduce_max(tf.concat(gate_filtern, 0), 0)
		#m = 1
	def g1(gate1,gate_u,gate_pu):
		return gate1*((1-gate_u)+gate_pu)
	def g2(gate1):
		return tf.ones(tf.shape(gate1),dtype=tf.float32)

	for k in range(filter_size*2+1):
		x_line = x_conv[:,k,:] #[batch, weigth]
		#W_filter_sum = tf.matmul(x_line, W[k,:,:], a_is_sparse = True, name = "W_filter_sum") #[batch, num_filters]
		
		if k < filter_size:
			gate_filter_contextp = gate_filterp[k][0,:,:] #1 is normal
			gate_filter_contextn = gate_filtern[k][0,:,:] #1 is anomaly
			gate_cu = 1 - (gate_filter_contextp + gate_filter_contextn) #1 is ambiguous
			#print gate
			#print m
			#gate2 = 1 - gate_filter_contextp
			#gate = gate1*(1-gate_u)
			#gate1*((1-gate_u)+gate_pu)
			#print g2(gate1)	
			#gate = tf.case({is_nogate: lambda:g2(gate1)}, default=lambda:g1(gate1,gate_u,gate_pu))
			#gate = gate1
			gatep = gate_filter_contextp*gate_pu + gate_u*gate_cu + gate_pu * gate_cu + gate_filter_contextp*gate_u
			gaten = gate_filter_contextn*gate_nu + gate_u*gate_cu + gate_nu * gate_cu + gate_filter_contextn*gate_u
			#gatep = gate_filter_contextp*gate_pu + gate_pu * gate_cu + gate_filter_contextp*gate_u
			#gaten = gate_filter_contextn*gate_nu + gate_nu * gate_cu + gate_filter_contextn*gate_u
			W_filter_sum_p = tf.matmul(x_line*gatep, W[k,:,:], a_is_sparse = True)
			W_filter_sum_n = tf.matmul(x_line*gaten, W[k,:,:], a_is_sparse = True)
			W_sum_p += W_filter_sum_p
			W_sum_n += W_filter_sum_n
			#W_sum_p += W_filter_sum*gatep
			#W_sum_n += W_filter_sum*gaten
			gate_kp.append(tf.expand_dims(gatep,0)) #[1, batch]
			gate_kn.append(tf.expand_dims(gaten,0))

		elif k > filter_size:
			gate_filter_contextp = gate_filterp[k-1][0,:,:]
			gate_filter_contextn = gate_filtern[k-1][0,:,:]
			gate_cu = 1 - (gate_filter_contextp + gate_filter_contextn)

			#gate1 = gate_filter_contextp + gate_filter_contextn
			#gate2 = 1 - gate_filter_contextp
			#gate = gate1*(1-gate_u)
			#gate = gate1
			#gate = gate1*((1-gate_u)+gate_pu)
			#gate = tf.case({is_nogate: lambda:g2(gate1)}, default=lambda:g1(gate1,gate_u,gate_pu))
			gatep = gate_filter_contextp*gate_pu + gate_u*gate_cu + gate_pu * gate_cu + gate_filter_contextp*gate_u
			gaten = gate_filter_contextn*gate_nu + gate_u*gate_cu + gate_nu * gate_cu + gate_filter_contextn*gate_u
			#gatep = gate_filter_contextp*gate_pu + gate_pu * gate_cu + gate_filter_contextp*gate_u
			#gaten = gate_filter_contextn*gate_nu + gate_nu * gate_cu + gate_filter_contextn*gate_u
			W_filter_sum_p = tf.matmul(x_line*gatep, W[k,:,:], a_is_sparse = True)
			W_filter_sum_n = tf.matmul(x_line*gaten, W[k,:,:], a_is_sparse = True)
			W_sum_p += W_filter_sum_p
			W_sum_n += W_filter_sum_n
			#W_sum_p += W_filter_sum*gatep
			#W_sum_n += W_filter_sum*gaten
			gate_kp.append(tf.expand_dims(gatep,0)) #[1, batch]
			gate_kn.append(tf.expand_dims(gaten,0))
		else:
			W_filter_sum_c = tf.matmul(x_line, W[k,:,:], a_is_sparse = True)
			W_sum_p += W_filter_sum_c #[batch, num_filters]
			W_sum_n += W_filter_sum_c

	convp = tf.add(W_sum_p, 0, name="convp") #[batch, num_filters]
	convn = tf.add(W_sum_n, 0, name="convn")
	#gNetp_1 = tf.concat(gNetp_temp1,0)  #[filter_heigth, batch, 2]
	#gNetn_1 = tf.concat(gNetn_temp1,0)
	gate_vp = tf.concat(gate_kp,0,name=namep) #[filter_heigth,batch]
	gate_vn = tf.concat(gate_kn,0,name=namen)

	return convp, convn, gate_vp, gate_vn
