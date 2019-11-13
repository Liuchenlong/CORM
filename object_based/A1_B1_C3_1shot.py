# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import sys, traceback
import pdb
import time
import os
import json
import numpy as np
import pickle
import random
import gc
import cv2
from utils import *
from PIL import Image
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
imgsize = 224
PiLImageResize = lambda x: x.resize((imgsize,imgsize))

def normalize(inp, activation, reuse, scope):
	return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
	""" Perform, conv, batch norm, nonlinearity, and max pool """
	stride, no_stride = [1,2,2,1], [1,1,1,1]
	conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
	normed = normalize(conv_output, activation, reuse, scope)
	normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
	return normed
def xent(pred, label):
	return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)

class GCNMetaModel(object):
	@classmethod
	def default_params(cls):
		return {
			'lr_decay_epoch': 5,
			'learning_rate': 0.005,
			'out_layer_dropout_keep_prob': 0.5,
			'edge_dims': 512,
			'hidden_size': 512,
			'num_timesteps': 2,
			'n_cluster': 16,
			'random_seed': 0,
		}
	def __init__(self, myconfig):
		args=[]
		self.args = args
		params = self.default_params()

		if myconfig is not None:
			params.update(myconfig)
		self.params = params

		self.train_dir = '/home/cll/fewshotlearning/GCN_feature/'
		#self.img_dir = '/home/cll/SUN397_224/'
		self.img_dir = '/home/0_public_data/MIT67/Images'
		#self.roi_dir = self.train_dir + 'simple/sunrois30v3'
		self.roi_dir = self.train_dir + 'simple/rois30'

		#self.train_data = json.load(open('sun_oneshot_train.json','r'))
		self.train_data = json.load(open('mit_oneshot_train.json','r'))
		#self.test_data = json.load(open('sun_oneshot_test.json','r'))
		self.test_data = json.load(open('mit_oneshot_test.json','r'))

		random.seed(params['random_seed'])
		np.random.seed(params['random_seed'])

		# Build the actual model
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		tf.set_random_seed(params['random_seed'])

		self.placeholders = {}
		self.ops = {}
		self.update_lr = 0.01
		self.num_updates= 5
		self.meta_lr = tf.placeholder_with_default(0.001, ())
		
		self.test_num_updates = 1 #train-1,test-10
		self.meta_batch_size = 2 
		self.update_batch_size = 1 #1shot-1,5shot-5
		self.loss_func = xent
		self.classification = True
		self.dim_hidden = 64
		self.channels = 3
		self.img_size = imgsize
		self.forward = self.forward_conv
		self.weights,self.weights2 = self.construct_conv_weights()
		self.prepare_specific_graph_model()
		self.make_model()
		self.initialize_model()
		
	def define_adj_salient_region(self, ROIS):
		r=[]
		r.append([imgsize*1.0,imgsize*1.0,0.,0.,imgsize*1.0,imgsize*1.0])
		for i in range(self.params['n_cluster']):
			r.append(ROIS[i])
		r=np.array(r)
		rois=r
		nc, _ = r.shape
		spatial_feats = np.zeros([nc, 5], dtype=np.float32)
		spatial_feats[:, [0, 2]] = rois[:, [2, 4]] / rois[:, 0][:, None]
		spatial_feats[:, [1, 3]] = rois[:, [3, 5]] / rois[:, 1][:, None]
		spatial_feats[:, 4] = ((rois[:, 4] - rois[:, 2]) * (rois[:, 5] - rois[:, 3])) /(rois[:, 0] * rois[:, 1])
		sfeats1 = np.tile(spatial_feats[:, None, :], [1, nc, 1])
		sfeats2 = np.tile(spatial_feats[None, :, :], [nc, 1, 1])
		_adjm = np.concatenate([sfeats1, sfeats2], axis=-1)
		return _adjm
		
	def initialize_model(self):
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(init_op)
		trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		print(trainable_vars)
	
	def prepare_specific_graph_model(self):
		self.placeholders['support_x'] = tf.placeholder(tf.float32,[self.meta_batch_size,5,imgsize,imgsize,3],name='support_x')
		self.placeholders['support_roi'] = tf.placeholder(tf.float32,[self.meta_batch_size,5,self.params['n_cluster'],6],name='support_roi')
		self.placeholders['support_adj'] = tf.placeholder(tf.float32,[self.meta_batch_size,5,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='support_adj')
		self.placeholders['target_x'] = tf.placeholder(tf.float32,[self.meta_batch_size,50,imgsize,imgsize,3],name='target_x')
		self.placeholders['target_roi'] = tf.placeholder(tf.float32,[self.meta_batch_size,50,self.params['n_cluster'],6],name='target_roi')
		self.placeholders['target_adj'] = tf.placeholder(tf.float32,[self.meta_batch_size,50,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='target_adj')
		self.placeholders['num_vertices'] = tf.placeholder(tf.int32, None)

		self.placeholders['support_label'] = tf.placeholder(tf.int32, [self.meta_batch_size,5])
		self.placeholders['target_label'] = tf.placeholder(tf.int32, [self.meta_batch_size,50])
		self.placeholders['support_v'] = tf.placeholder(tf.int32, [self.meta_batch_size,5])
		self.placeholders['target_v'] = tf.placeholder(tf.int32, [self.meta_batch_size,50])
		
		self.placeholders['is_training'] = tf.placeholder(tf.bool)
	
	def construct_conv_weights(self):
		weights = {}
		weights2 = {}

		dtype = tf.float32
		conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
		fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
		k = 3
		
		weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
		weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
		weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
		weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
		weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
		weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
		weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
		weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
		
		weights['w5'] = tf.get_variable('w5', [self.params['edge_dims'], 1], initializer=fc_initializer,)
		
		weights['w6'] = tf.get_variable('w6', [int(self.params['in_size']), self.params['hidden_size']], initializer=fc_initializer)
		
		weights['w7'] = tf.get_variable('w7', [self.params['hidden_size'], 5], initializer=fc_initializer)
		weights['b7'] = tf.Variable(tf.zeros([5]), name='b7')
		weights['w72'] = tf.get_variable('w72', [64*14*14, 5], initializer=fc_initializer)
		weights['b72'] = tf.Variable(tf.zeros([5]), name='b72')
		
		return weights,weights2
	
	def forward_conv(self, input, weights, rois, edge, reuse=tf.AUTO_REUSE, scope=''):
		
		input = tf.reshape(input, [-1, imgsize, imgsize, 3])
		rois = tf.reshape(rois, [-1, self.params['n_cluster'], 6])
		
		
		hidden1 = conv_block(input, weights['conv1'], weights['b1'], reuse, scope+'0')
		hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
		hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
		hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
		
		gpool5 = tf.reduce_mean(hidden4,axis=[1,2])
		num_bbox = rois.shape.as_list()[1]
		batch_size = rois.shape.as_list()[0]
		roi_pool5 = roi_pool_layer(hidden4, rois, 7, batch_size, name='ROIPooling')
		roi_conv = global_pool(roi_pool5, name='roi_global_pool', ptype='max')
		roi_conv = tf.reshape(tf.squeeze(roi_conv), [batch_size, num_bbox, -1])
		hidden5 = tf.concat([gpool5[:, None, :], roi_conv], axis=1)
		
		v = self.placeholders['num_vertices']
		h_dim = self.params['hidden_size']
		e_dim = self.params['edge_dims']
		o_dim = self.params['out_size']

		h = tf.nn.l2_normalize(hidden5, axis=-1)
		self.norm_init_h = h
		in_dim = self.params['in_size']
		init_h = tf.pad(h, [[0, 0], [0, 0], [0, o_dim-in_dim]])

		big_number = 1e12
		diag_I = tf.diag(tf.ones([v]))
		diag_I = tf.stop_gradient(diag_I)
		mask = tf.stop_gradient(tf.ones([v, v]) - tf.diag(tf.ones([v])))
		inf_mask = big_number * tf.diag(tf.ones([v]))
		inf_mask = tf.stop_gradient(inf_mask)
		
		# for edge graph
		e = edge
		print e.shape
		e = tf.concat(e, axis=0)
		e = tf.reshape(e, [-1, e_dim])
		print e.shape
		a = tf.matmul(e, weights['w5'])
		a = tf.reshape(a, [-1, v, v])
		a -= inf_mask[None, :, :]
		a = tf.nn.softmax(a, axis=-1)
		a = a + diag_I[None, :, :]
		d = tf.matrix_diag(1 / tf.reduce_sum(a, axis=-1))
		sa = tf.matmul(d, a)
		for i in range(self.params['num_timesteps']):
			sah = tf.matmul(sa, h)
			sah = tf.reshape(sah, [-1, in_dim])
			smh = tf.nn.relu(tf.matmul(sah, weights['w6']))
			smh = tf.reshape(smh, [-1, v, h_dim])
			h = smh
			#h = tf.nn.l2_normalize(h, axis=-1)
		h += init_h
		
		global_node = h[:,0,:]
		local_nodes = h[:,1:,:]
		print(global_node.shape)
		possibility = tf.matmul(global_node, weights['w7']) + weights['b7']
		#possibility2 = tf.matmul(tf.reshape(local_nodes,[-1,128]), weights['w72']) + weights['b72']
		possibility2 = tf.matmul(tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])]), weights['w72']) +weights['b72']
		return possibility,possibility2
	
	def task_metalearn(self, inp, reuse=tf.AUTO_REUSE):
		""" Perform gradient descent for one task in the meta-batch. """
		inputa, inputb, roia, roib, labela, labelb, edgea, edgeb = inp
		task_outputbs, task_lossesb = [], []
		num_updates = self.num_updates
		task_accuraciesb = []
		task_outputa,local_output = self.forward(inputa, self.weights, roia, edgea, reuse=reuse)  # only reuse on the first iter
		#task_lossa = (self.loss_func(task_outputa, tf.one_hot(labela,5)) + self.params['node_lambda'] * tf.reduce_mean(tf.reshape(self.loss_func(local_output, tf.tile(tf.one_hot(labela,5)[:,None,:],[1,self.params['n_cluster'],1])),(-1,30)),1))/ self.update_batch_size
		task_lossa = (self.loss_func(task_outputa, tf.one_hot(labela,5)) + self.loss_func(local_output, tf.one_hot(labela,5)))/ self.update_batch_size
		
		
		grads = tf.gradients(task_lossa, list(self.weights.values()))
		gradients = dict(zip(self.weights.keys(), grads))
		fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr*gradients[key] for key in self.weights.keys()]))
		output,local_output = self.forward(inputb, fast_weights, roib, edgeb, reuse=True)
		task_outputbs.append(output)
		#task_lossesb.append((self.loss_func(output, tf.one_hot(labelb,5)) + self.params['node_lambda'] * tf.reduce_mean(tf.reshape(self.loss_func(local_output, tf.tile(tf.one_hot(labelb,5)[:,None,:],[1,self.params['n_cluster'],1])),(-1,30)),1))/ self.update_batch_size)
		task_lossesb.append((self.loss_func(output, tf.one_hot(labelb,5)) + self.loss_func(local_output, tf.one_hot(labelb,5)))/ self.update_batch_size)
		for j in range(num_updates - 1):
			task_outputa,local_output = self.forward(inputa, fast_weights, roia, edgea, reuse=reuse)
			#loss = (self.loss_func(task_outputa, tf.one_hot(labela,5)) + self.params['node_lambda'] * tf.reduce_mean(tf.reshape(self.loss_func(local_output, tf.tile(tf.one_hot(labela,5)[:,None,:],[1,self.params['n_cluster'],1])),(-1,30)),1))/ self.update_batch_size		
			loss = (self.loss_func(task_outputa, tf.one_hot(labela,5)) + self.loss_func(local_output, tf.one_hot(labela,5)))/ self.update_batch_size
			grads = tf.gradients(loss, list(fast_weights.values()))
			gradients = dict(zip(fast_weights.keys(), grads))
			fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
			output,local_output = self.forward(inputb, fast_weights, roib, edgeb, reuse=True)
			task_outputbs.append(output)
			#task_lossesb.append((self.loss_func(output, tf.one_hot(labelb,5)) + self.params['node_lambda'] * tf.reduce_mean(tf.reshape(self.loss_func(local_output, tf.tile(tf.one_hot(labelb,5)[:,None,:],[1,self.params['n_cluster'],1])),(-1,30)),1))/ self.update_batch_size)
			task_lossesb.append((self.loss_func(output, tf.one_hot(labelb,5)) + self.loss_func(local_output, tf.one_hot(labelb,5)))/ self.update_batch_size)
		
		task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
		task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(tf.one_hot(labela,5), 1))
		for j in range(num_updates):
			task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(tf.one_hot(labelb,5), 1)))
		task_output.extend([task_accuracya, task_accuraciesb])
		
		return task_output
	
	def make_model(self, prefix='metatrain_'):
		inputa = self.placeholders['support_x']
		inputb = self.placeholders['target_x']
		roia = self.placeholders['support_roi']
		roib = self.placeholders['target_roi']
		labela = self.placeholders['support_label']
		labelb = self.placeholders['target_label']
		edgea = self.placeholders['support_adj']
		edgeb = self.placeholders['target_adj']
		with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
			# outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
			lossesa, outputas, lossesb, outputbs = [], [], [], []
			accuraciesa, accuraciesb = [], []
			num_updates = self.num_updates
			outputbs = [[]]*num_updates
			lossesb = [[]]*num_updates
			accuraciesb = [[]]*num_updates
			
			unused = self.task_metalearn((inputa[0], inputb[0], roia[0], roib[0], labela[0], labelb[0], edgea[0], edgeb[0]), tf.AUTO_REUSE)
			
			out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
			out_dtype.extend([tf.float32, [tf.float32]*num_updates])
			result = tf.map_fn(self.task_metalearn, elems=(inputa, inputb, roia, roib, labela, labelb, edgea, edgeb), dtype=out_dtype, parallel_iterations=self.meta_batch_size)
			outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
			
			##TRAIN
			## Performance & Optimization
			self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
			self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
			# after the map_fn
			self.outputas, self.outputbs = outputas, outputbs
			self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
			self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
			self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
			optimizer = tf.train.AdamOptimizer(self.meta_lr)
			self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.num_updates-1])
			for grad,var in gvs:
				print grad,var
			gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
			self.metatrain_op = optimizer.apply_gradients(gvs)
			##TEST
			self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
			self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
			self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
			self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(num_updates)]
			


	def get_data_batch(self, is_training):
		support_set = []
		target_set = []
		if is_training:
			data = self.train_data
		else:
			data = self.test_data
		for batch in range(self.meta_batch_size):
			selected_classes = np.random.choice(len(data.keys()), 5, False)
			support_x=[]
			for k,classes in enumerate(selected_classes):
				dirs = data[data.keys()[classes]] 
				roi_dirs = [os.path.join(self.roi_dir, '_'.join(dir.split()[0][:-4].split('/')) + '.npy') for dir in dirs]
				img_dirs = [os.path.join(self.img_dir, dir) for dir in dirs]
				selected_samples = np.random.choice(len(dirs), 1+10, False)
				selected_roi = [roi_dirs[i] for i in selected_samples]
				selected_img = [img_dirs[i] for i in selected_samples]
				roi_dirs = [np.load(dir) for dir in selected_roi]
				img_dirs = [np.array(PiLImageResize(filenameToPILImage(dir)))*1.0/255.0 for dir in selected_img]
				support_x.extend([(img_dirs[i], roi_dirs[i], self.define_adj_salient_region(roi_dirs[i]), k) for i in range(len(roi_dirs))[:1]])
				target_set.extend([(img_dirs[i], roi_dirs[i], self.define_adj_salient_region(roi_dirs[i]), k) for i in range(len(roi_dirs))[1:]])
				np.random.shuffle(support_x)
			support_img,support_roi,support_adj,support_y = zip(*support_x)
			support_set.append((support_img, support_roi, support_adj, support_y))
		support_img,support_roi,support_adj,support_y = zip(*support_set)

		support_img = np.reshape(np.array(support_img), (self.meta_batch_size, 5, imgsize, imgsize, 3))
		support_roi = np.reshape(np.array(support_roi), (self.meta_batch_size, 5, self.params['n_cluster'], 6))
		support_adj = np.reshape(np.array(support_adj), (self.meta_batch_size, 5, self.params['n_cluster']+1, self.params['n_cluster']+1, self.params['edge_dims']))
		support_set_y = np.reshape(np.array(support_y), (self.meta_batch_size, 5))
		support = {'img':support_img, 'roi':support_roi, 'adj':support_adj}
		target_img,target_roi,target_adj,target_y = zip(*target_set)
		target_img = np.reshape(np.array(target_img), (self.meta_batch_size, 50, imgsize, imgsize, 3))
		target_roi = np.reshape(np.array(target_roi), (self.meta_batch_size, 50, self.params['n_cluster'], 6))
		target_adj = np.reshape(np.array(target_adj), (self.meta_batch_size, 50, self.params['n_cluster']+1, self.params['n_cluster']+1, self.params['edge_dims']))
		target_set_y = np.reshape(np.array(target_y), (self.meta_batch_size, 50))
		target = {'img':target_img, 'roi':target_roi, 'adj':target_adj}
		return support, support_set_y, target, target_set_y


	def run_epoch(self, is_training):

		loss = 0
		start_time = time.time()

		step = 0
		fpath = []
		if is_training:
			epo = 1
		else:
			epo = 400
			metaval_accuracies = []
		for epoch in range(epo):
			#support_x : dict{'feat':,'adjm':}
			#print epoch
			support_x, support_y, target_x, target_y = self.get_data_batch(is_training)
			num_node = support_x['roi'].shape[1]
			#print(support_x['adj'].shape)
			batch_data = {
				self.placeholders['support_x']: support_x['img'],
				self.placeholders['support_roi']: support_x['roi'],
				self.placeholders['support_adj']: support_x['adj'],
				self.placeholders['support_label']: support_y,
				self.placeholders['target_x']: target_x['img'],
				self.placeholders['target_roi']: target_x['roi'],
				self.placeholders['target_adj']: target_x['adj'],
				self.placeholders['target_label']: target_y,
				self.placeholders['num_vertices']: self.params['n_cluster'] + 1,
			}

			if is_training:
				batch_data[self.placeholders['is_training']] = True
			else:
				batch_data[self.placeholders['is_training']] = False
			
			if is_training:
				input_tensors = [self.metatrain_op, self.total_loss1, self.total_losses2[self.num_updates-1], self.total_accuracy1, self.total_accuracies2[self.num_updates-1]]
				result = self.sess.run(input_tensors, feed_dict=batch_data)
				return np.mean(result[-4]),np.mean(result[-3]),np.mean(result[-2]),np.mean(result[-1])
			else:
				input_tensors = [[self.metaval_total_accuracy1] + self.metaval_total_accuracies2]
				result = self.sess.run(input_tensors, feed_dict=batch_data)
				metaval_accuracies.append(result)
			
		metaval_accuracies = np.array(metaval_accuracies)
		means = np.mean(metaval_accuracies, 0)
		stds = np.std(metaval_accuracies, 0)
		ci95 = 1.96*stds/np.sqrt(400)
		
		return means,stds,ci95
