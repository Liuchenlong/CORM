# -*- coding: utf-8 -*-
import tensorflow as tf
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


class RelationModule:
    def __init__(self):
        self.reuse = False
    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        this module use to implement relation module
        """
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        with tf.variable_scope('RelationModule', reuse=self.reuse):
            with tf.variable_scope('conv_layers'):
                with tf.variable_scope('RelationModule_conv1'):
                    g_conv1_encoder = tf.layers.conv1d(image_input, 64, 9, strides=1, padding='VALID')
                    g_conv1_encoder = leaky_relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.layers.max_pooling1d(g_conv1_encoder, 2, 2, padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)
                with tf.variable_scope('RelationModule_conv2'):
                    g_conv2_encoder = tf.layers.conv1d(g_conv1_encoder, 64, 9, strides=1, padding='VALID')
                    g_conv2_encoder = leaky_relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.layers.max_pooling1d(g_conv2_encoder, 2, 2, padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)
                with tf.variable_scope('fully_connected_relu'):
                    g_fc1_encoder = tf.contrib.layers.flatten(g_conv2_encoder)
                    g_fc1_encoder = tf.contrib.layers.fully_connected(g_fc1_encoder, 8, trainable=True, scope='fc_relu')
                with tf.variable_scope('fully_connected_sigmoid'):
                    g_fc2_encoder = tf.contrib.layers.fully_connected(g_fc1_encoder, 1, activation_fn=tf.nn.sigmoid,
                                                                      trainable=True, scope='fc_sigmoid')
            g_conv_encoder = g_fc2_encoder

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RelationModule')
        return g_conv_encoder

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
        #myconfig = None
        self.args = args
        # Collect parameters:
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
        self.weights = {}
        self.ops = {}
        self.RelationModule = RelationModule()
        self.make_model()
        self.make_train_step()
        self.initialize_model()


    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)



    def prepare_specific_graph_model(self):
        self.placeholders['support_x'] = tf.placeholder(tf.float32,[5,224,224,3],name='support_x')
        self.placeholders['support_roi'] = tf.placeholder(tf.float32,[5,self.params['n_cluster'],6],name='support_roi')
        self.placeholders['support_adj'] = tf.placeholder(tf.float32,[5,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='support_adj')
        self.placeholders['target_x'] = tf.placeholder(tf.float32,[50,224,224,3],name='target_x')
        self.placeholders['target_roi'] = tf.placeholder(tf.float32,[50,self.params['n_cluster'],6],name='target_roi')
        self.placeholders['target_adj'] = tf.placeholder(tf.float32,[50,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='target_adj')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, None)
        self.placeholders['support_label'] = tf.placeholder(tf.int32, [5])
        self.placeholders['target_label'] = tf.placeholder(tf.int32, [50])
        self.placeholders['support_v'] = tf.placeholder(tf.int32, [5])
        self.placeholders['target_v'] = tf.placeholder(tf.int32, [50])
        self.placeholders['is_training'] = tf.placeholder(tf.bool)
    def define_adj_salient_region(self, ROIS):
    # methods from visual relationship Detection with Internal and External Knowledge Distillation
        r=[]
        r.append([224.,224.,0.,0.,224.,224.])
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

    def get_feature(self, X, rois, edge, is_training=False):
        output = {}
        print('############################')
        # Layer 1 
        x = conv(X, 3, 64, 1, pad=1, name='clf1_conv', with_bias=True)
        output['conv1'] = x
        x = BatchNorm(x, is_training, name='clf1_bn')
        x = tf.nn.leaky_relu(x)
        output['bn1'] = x
        print(x.shape)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)

        # Layer 2 
        x = conv(x, 3, 64, 1, pad=1, name='clf2_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf2_bn')
        x = tf.nn.leaky_relu(x)
        print(x.shape)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)

        # Layer 3 
        x = conv(x, 3, 64, 1, pad=1, name='clf3_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf3_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)

        # Layer 4 
        layer4 = conv(x, 3, 64, 1, pad=1, name='clf4_conv', with_bias=True)
        layer4 = BatchNorm(layer4, is_training, name='clf4_bn')
        layer4 = tf.nn.leaky_relu(layer4)
        layer4 = max_pool(layer4, 2, 2, padding='SAME')
        print(layer4.shape)
        output['gpool5'] = tf.reduce_mean(layer4, axis=[1, 2])
        # ROI pooling layer
        num_bbox = rois.shape.as_list()[1]
        batch_size = rois.shape.as_list()[0]
        roi_pool5 = roi_pool_layer(layer4, rois, 7, batch_size, name='ROIPooling')
        roi_conv = global_pool(roi_pool5, name='roi_global_pool', ptype='max')
        roi_conv = tf.reshape(tf.squeeze(roi_conv), [batch_size, num_bbox, -1])
        output['roi_conv'] = roi_conv
        
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        e_dim = self.params['edge_dims']
        o_dim = self.params['out_size']
        h = tf.concat([output['gpool5'][:, None, :], output['roi_conv']], axis=1)  # [b, v, h]
        print(h.shape)

        h = tf.nn.l2_normalize(h, axis=-1)
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
        e = tf.concat(e, axis=0)
        e = tf.reshape(e, [-1, e_dim])
        a = fc(e, 1, 'proj_e', with_bias=False, relu=False, reuse=tf.AUTO_REUSE)
        a = tf.reshape(a, [-1, v, v])
        a -= inf_mask[None, :, :]
        a = tf.nn.softmax(a, axis=-1)
        a = a + diag_I[None, :, :]
        d = tf.matrix_diag(1 / tf.reduce_sum(a, axis=-1))
        sa = tf.matmul(d, a)
        for i in range(self.params['num_timesteps']):
            sah = tf.matmul(sa, h)
            sah = tf.reshape(sah, [-1, in_dim])
            smh = fc(sah, h_dim, 'gcn_{}_sim'.format(i), with_bias=False, relu=True, reuse=tf.AUTO_REUSE)
            smh = tf.reshape(smh, [-1, v, h_dim])
            h = smh
            #h = tf.nn.l2_normalize(h, axis=-1)
        h += init_h
        return h


    def make_model(self):
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')
        self.placeholders['relation_keep_prob'] = tf.placeholder(tf.float32, None, name='relation_keep_prob')
        with tf.variable_scope("graph_model", reuse = tf.AUTO_REUSE):
            self.prepare_specific_graph_model()
            support_final_node_representations = self.get_feature(self.placeholders['support_x'],self.placeholders['support_roi'],self.placeholders['support_adj'],self.placeholders['is_training'])
            target_final_node_representations = self.get_feature(self.placeholders['target_x'],self.placeholders['target_roi'],self.placeholders['target_adj'],self.placeholders['is_training'])
        with tf.variable_scope("out_layer", reuse = tf.AUTO_REUSE):
            with tf.variable_scope("regression_gate"):
                self.weights['regression_node'] = MLP(self.params['out_size'], 5, [],
                                                      self.placeholders['out_layer_dropout_keep_prob'])
        node_loss = tf.constant(0.0)
        print('Node supervision')
        tv = self.placeholders['num_vertices']-1
        v = self.placeholders['num_vertices']
        support_node = support_final_node_representations[:, 1:, :]
        node_loss = []
        for i in range(5):
            last_h1 = tf.reshape(support_node[i], [-1, self.params['out_size']])
            node_out = self.weights['regression_node'](last_h1)
            node_out = tf.reshape(node_out, [-1, 5])
            node_labels = tf.tile(tf.reshape(self.placeholders['support_label'][i],(1,1)),[1, tv])
            print(node_labels.get_shape().as_list())
            node_labels = tf.reshape(node_labels, [-1])
            node_loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node_labels,logits=node_out))
        target_node = target_final_node_representations[:, 1:, :]
        for i in range(50):
            last_h1 = tf.reshape(target_node[i], [-1, self.params['out_size']])
            node_out = self.weights['regression_node'](last_h1)
            node_out = tf.reshape(node_out, [-1, 5])
            node_labels = tf.tile(tf.reshape(self.placeholders['target_label'][i],(1,1)),[1, tv])
            node_labels = tf.reshape(node_labels, [-1])
            node_loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node_labels,logits=node_out))
        node_loss = tf.reduce_mean(tf.concat(node_loss,0)) * self.params['node_lambda']
        support_final_node_representations = tf.reshape(support_final_node_representations[:,0,:],[5,1,self.params['out_size']])
        target_final_node_representations = tf.reshape(target_final_node_representations[:,0,:],[50,1,self.params['out_size']])
        concat_node_representations = tf.concat([tf.stack([support_final_node_representations] * 50), tf.stack([target_final_node_representations] * 5, axis=1)],2)
        print("concat_node_representations ",concat_node_representations.get_shape().as_list())
        concat_node_representations = tf.transpose(concat_node_representations, [0,1,3,2])
        print("concat_node_representations ",concat_node_representations.get_shape().as_list())
        [num_query, num_classes, dim_1, dim_2] = concat_node_representations.get_shape().as_list()
        concat_node_representations = tf.reshape(concat_node_representations, [50*5, dim_1, dim_2])
        similarities = self.RelationModule(concat_node_representations, training=self.placeholders['is_training'], keep_prob=self.placeholders['relation_keep_prob'])
        print("similarities ",similarities.get_shape().as_list())   
        similarities = tf.reshape(similarities, [50, 5])         
        support_set_labels = tf.one_hot(self.placeholders['support_label'], 5)
        preds = tf.squeeze(tf.matmul(similarities, support_set_labels))
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.placeholders['target_label'], tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        targets = tf.one_hot(self.placeholders['target_label'], 5)
        print("preds ",preds.get_shape().as_list())   
        print("targets ",targets.get_shape().as_list())   
        loss = tf.reduce_mean(tf.square((preds-1)*targets + preds*(1-targets)))
        self.ops['accuracy'] = accuracy
        self.ops['loss'] = loss + node_loss

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.lr = tf.Variable(self.params['learning_rate'], trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        print(trainable_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):  # Needed for correct batch norm usage
            c_error_opt_op = optimizer.minimize(self.ops['loss'], var_list=trainable_vars)
        
        self.ops['train_step'] = c_error_opt_op


    def get_data_batch(self, is_training):
        support_set = []
        target_set = []
        if is_training:
            data = self.train_data
        else:
            data = self.test_data
        for batch in range(1):
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

        support_img = np.array([img for img in support_img[0]])
        support_roi = np.array([img for img in support_roi[0]])
        support_adj = np.array([img for img in support_adj[0]])
        support_set_y = [_lbl for _lbl in zip(*support_y)]
        support = {'img':support_img, 'roi':support_roi, 'adj':support_adj}
        target_img,target_roi,target_adj,target_y = zip(*target_set)
        target_img = np.array([img for img in target_img])
        target_roi = np.array([img for img in target_roi])
        target_adj = np.array([img for img in target_adj])
        target_set_y = target_y
        target = {'img':target_img, 'roi':target_roi, 'adj':target_adj}
        return support, np.reshape(support_set_y,(-1)), target, np.reshape(target_set_y,(-1))


    def run_epoch(self, is_training):

        loss = 0
        accuracy = []
        accuracy_op = self.ops['accuracy']
        start_time = time.time()

        step = 0
        fpath = []
        if is_training:
            epo = 400
        else:
            epo = 400
        for epoch in range(epo):
            support_x, support_y, target_x, target_y = self.get_data_batch(is_training)
            num_node = support_x['roi'].shape[1]
            batch_data = {
                self.placeholders['support_x']: support_x['img'],
                self.placeholders['support_roi']: support_x['roi'],
                self.placeholders['support_adj']: support_x['adj'],
                self.placeholders['support_label']: support_y,
                self.placeholders['target_x']: target_x['img'],
                self.placeholders['target_roi']: target_x['roi'],
                self.placeholders['target_adj']: target_x['adj'],
                self.placeholders['target_label']: target_y,
                self.placeholders['num_vertices']: num_node + 1,
            }

            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                batch_data[self.placeholders['is_training']] = True
                batch_data[self.placeholders['relation_keep_prob']] = 0.5
                fetch_list = [self.ops['loss'], accuracy_op, self.ops['train_step']]

            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                batch_data[self.placeholders['is_training']] = False
                batch_data[self.placeholders['relation_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_op]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracy) = (result[0], result[1])
            loss += batch_loss
            accuracy.append(batch_accuracy)

        acc = np.mean(accuracy)
        std = np.std(accuracy, 0)
        ci95 = 1.96*std/np.sqrt(epo)
        loss = loss / epo
        instance_per_sec = epo / (time.time() - start_time)
        return loss, acc, instance_per_sec, ci95

