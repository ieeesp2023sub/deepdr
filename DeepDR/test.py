# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
os.environ["PBR_VERSION"]='3.1.1'
import sys
import itertools
import data_helpers
from CNN import CNN
from Queue import Queue
import networkx as nx

def testing(basepath, filter_size, num_dim):

	if not os.path.exists('./output'):
		os.makedirs('./output')

	fs = '1'
	BATCH_SIZE = 1
	NUM_CLASSES = 2
	#FILTER_SIZES = int(sys.argv[2])
	FILTER_SIZES = int(filter_size)
	USERNAME = os.listdir(basepath)
	#LENGTH = 201
	POSITION_RATE = 0.8
	NUM_DIM = int(num_dim)

	MODEL_SAVE_PATH = "./to/model/"
	MODEL_NAME = "model.ckpt"

	data_list = []
	label_list = []
	date_list = []
	user_list = []
	labelcontext_list = []
	datecontext_list = []

	testfile = open('./testfile.csv')
	labeltestfile = open('./testlabel.csv')
	data_path_list = [i.rstrip() for i in testfile.readline()[0:-1].split(',')]
	label_path_list = [i.rstrip() for i in labeltestfile.readline()[0:-1].split(',')]

	for index, pathi in enumerate(data_path_list):
		name = pathi.split('/')[2]
		data_i, label_i, date_i, l_i = data_helpers.data_load_1(pathi, label_path_list[index], NUM_CLASSES, name)
		#NUM_DIM = len(data_i[0])
		x_i, lc_i, dc_i = data_helpers.generating_x_l(np.array(data_i),np.array(l_i),np.array(date_i),FILTER_SIZES)
		data_list = data_list+x_i
		labelcontext_list = labelcontext_list+lc_i
		datecontext_list = datecontext_list+dc_i
		label_list = label_list+label_i
		date_list = date_list+date_i


	data = np.array(data_list)
	labelcontext = np.array(labelcontext_list)
	datecontext = np.array(datecontext_list)
	label = np.array(label_list)
	date = np.array(date_list)
	attackstory = open('./output/attackstory.txt','w')

	alertevents = []
	alertevents_date = []
	label_scores = []
	sum_a = 0
	sum_b = 0
	checkpoint_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			batch_size = graph.get_operation_by_name("batch_size").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			representation_b = graph.get_operation_by_name("h_p").outputs[0]
			representation_a = graph.get_operation_by_name("h_n").outputs[0]
			gatep = graph.get_operation_by_name("gatep").outputs[0]
			gaten = graph.get_operation_by_name("gaten").outputs[0]
			predictions = graph.get_operation_by_name("CNN_output/softmax").outputs[0]
			representation_b_input = graph.get_operation_by_name("representation_b").outputs[0]
			representation_a_input = graph.get_operation_by_name("representation_a").outputs[0]
			recontruction_error_b = graph.get_operation_by_name("autoencoderloss_b/autoencoderloss_b").outputs[0]
			recontruction_error_a = graph.get_operation_by_name("autoencoderloss_a/autoencoderloss_a").outputs[0]
		
			attack_num = 0
			benign_num = 0
			benign_detection_num = 0
			attack_detection_num = 0
			add_detection_num = 0
			alertid = 0

			benign_num_c = 0
			attack_num_c = 0
			benign_detection_num_c = 0
			attack_detection_num_c = 0
			fbenign_detection_num_c = 0
			fattack_detection_num_c = 0
			gt = {}	
		
			for i, x_line in enumerate(data):
				x_line_dim = []
				for x_line_i in x_line:
					weigth = [0]*NUM_DIM
					if x_line_i != 0:
						weigth[x_line_i-1] = 1
					x_line_dim.append(weigth)

				x_line = np.array([x_line_dim])
				predictions_show, representation_r_b, representation_r_a, gatep_r, gaten_r = sess.run([predictions, representation_b, representation_a, gatep, gaten], {input_x:x_line, batch_size: BATCH_SIZE, dropout_keep_prob: 1.0})
				recontruction_b = sess.run([recontruction_error_b],{representation_b_input:representation_r_b})
				recontruction_a = sess.run([recontruction_error_a],{representation_a_input:representation_r_a})
			
				if date[i].split('_')[0] != '0':
					
					labelcontext_i = np.r_[labelcontext[i][0:FILTER_SIZES],labelcontext[i][FILTER_SIZES+1:]]
					datecontext_i = np.r_[datecontext[i][0:FILTER_SIZES],datecontext[i][FILTER_SIZES+1:]]
				
					#print date[i], label[i], gatep_r[:,0,0], recontruction_b
					#print date[i], label[i], gaten_r[:,0,0], recontruction_a
					recontruction_b = recontruction_b[0]
					recontruction_a = recontruction_a[0]
					sum_a += recontruction_a
					sum_b += recontruction_b
					label_scores.append((date[i],label[i,0],recontruction_b,recontruction_a,x_line[0,FILTER_SIZES,:]))

	threshold_a = 1.0*sum_a/len(label_scores)-0.1
	threshold_b = 1.0*sum_b/len(label_scores)+0.1
	for score in label_scores:
		date = score[0]
		label = score[1]
		recontruction_b = score[2]
		recontruction_a = score[3]
		event = score[4]

		if label == 1:
			benign_num += 1
					
			if recontruction_a <= threshold_a or (recontruction_a > threshold_a and recontruction_b > threshold_b):
			#if recontruction_a <= threshold_a:
				alertid += 1
				alertevents.append((date+'_fal_'+str(alertid),event))
				#alertevents_false.append(date[i])
				date_i = date.split('_')[0]
				alertevents_date.append(date_i)	
						
				#if date[i] not in add_events:   
				benign_detection_num += 1

		else:
			attack_num += 1
			if recontruction_a <= threshold_a or (recontruction_a > threshold_a and recontruction_b > threshold_b):
			#if recontruction_a <= threshold_a:
				name = date.split('_')[1]
				date_i = date.split('_')[0]
				alertevents_date.append(date_i)
				if gt.has_key(name):
					gt[name].append(date_i)
				else:
					gt[name] = [date_i]
				alertid += 1
				attack_detection_num += 1
				alertevents.append((date+'_tur_'+str(alertid),event))
				#alertevents_true.append(date[i])
				
		
	MODELB_SAVE_PATH = "./to/modelB/"
	MODELB_NAME = "modelb.ckpt"
	checkpoint_file = tf.train.latest_checkpoint(MODELB_SAVE_PATH)
	graphB = tf.Graph()
	with graphB.as_default():
		sess = tf.Session()
		with sess.as_default():
		
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			cor_x = graphB.get_operation_by_name("cor_x").outputs[0]
			score_cp = graphB.get_operation_by_name("score_cp").outputs[0]
			#score_cn = graphB.get_operation_by_name("score_cn").outputs[0]

			pair_link = []
			for pair_i in itertools.combinations(alertevents, 2):
				pair_x = pair_i[0][1]+pair_i[1][1]
				pair_x = np.array([pair_x.tolist()])
				#s_cp, s_cn = sess.run([score_cp,score_cn], {cor_x:pair_x})
				s_cp = sess.run([score_cp], {cor_x:pair_x})
				if s_cp[0][0][1] > 0.99:
					#if pair_i[0][0].split('_')[1] == pair_i[1][0].split('_')[1]:
					t1 = int(pair_i[0][0].split('_')[-1])
					t2 = int(pair_i[1][0].split('_')[-1])
					if t1 < t2:
						pair_link.append([pair_i[0][0],pair_i[1][0],s_cp[0][0][1]])
						#print pair_i[0][0], pair_i[1][0], s_cp[0][0][1]
					else:
						pair_link.append([pair_i[1][0],pair_i[0][0],s_cp[0][0][1]])
						#print pair_i[1][0], pair_i[0][0], s_cp[0][0][1]
					#print pair_i[1][0], pair_i[0][0], s_cp[0][0][1]
		
			group_false_num = 0
			group = []
			graph = nx.Graph()
			dr = {}
			for edge in pair_link:
				graph.add_edge(edge[0],edge[1],score=edge[2])
			for c in nx.connected_components(graph):
				nodeSet = graph.subgraph(c).nodes()
				edgeSet = graph.subgraph(c).edges(data=True)
				score = 0.0
				for edge in edgeSet:
					score += edge[2]['score']
				score = score/len(edgeSet)

				if len(nodeSet) >=2:
					for e in nodeSet:
						if 'fal' in e:
							group_false_num += 1
				group.append((nodeSet,score))
				attackstory.write(str(list(nodeSet)))
				attackstory.write('\n')
				for node in nodeSet:
					#name = node.split('_')[1]
					name = 'ACM2278'
					date_i = node.split('_')[0]
					if dr.has_key(name):
						dr[name].append(date_i)
					else:
						dr[name] = [date_i]

	recall = 1.0*attack_detection_num / attack_num
	if attack_detection_num + benign_detection_num != 0:
		precise = 1.0*attack_detection_num / (attack_detection_num + benign_detection_num)
	else:
		precise = 0.0

	if attack_detection_num + group_false_num != 0:
		precise_group = 1.0*attack_detection_num / (attack_detection_num + group_false_num)
	else:
		precise_group = 0.0

	'''
	tc = Timecluster(alertevents_date,60*10)
	dr_t = tc.run()
	c_num = 0
	dr = {}
	for ci in dr_t:
		c_num+=1
		dr[str(c_num)] = ci
	print dr
	'''

	print 'recall: '+str(recall)+'; precise: '+str(precise_group)

