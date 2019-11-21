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
from sklearn.cluster import k_means, AgglomerativeClustering

filenameToPILImage = lambda x: Image.open(x).convert('RGB')
imgsize = 224
PiLImageResize = lambda x: x.resize((imgsize,imgsize))

def clamp(inputs, min_value=None, max_value=None):
    output = inputs
    if min_value is not None:
        output[output < min_value] = min_value
    if max_value is not None:
        output[output > max_value] = max_value
    return output

def logAndSign(inputs, k=5):
    eps = np.finfo(inputs.dtype).eps
    log = np.log(np.absolute(inputs) + eps)
    clamped_log = clamp(log / k, min_value=-1.0)
    sign = clamp(inputs * np.exp(k), min_value=-1.0, max_value=1.0)
    return np.concatenate([clamped_log, sign], axis=1)

def findSalientRegion(img, thresh):
    # iimg = (img > thresh).astype(np.uint8)
    _, iimg = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(iimg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([imgsize, imgsize, x, y, x+w, y+h])
    boxes = np.array(boxes)
    return boxes

def get_im_list(im_dir, roi_dir, dest_dir, file_path, cam_dir):
    im_list = []
    im_labels = []
    roi_path = []
    dest_path = []
    cam_path = []
    with open(file_path, 'r') as fi:
        for line in fi:
            im_list.append(os.path.join(im_dir, line.split()[0]))
            im_labels.append(int(line.split()[-1]))
            fnewname = '_'.join(line.split()[0][:-4].split('/'))
            roi_path.append(os.path.join(roi_dir, fnewname + '.npy'))
            dest_path.append(os.path.join(dest_dir, fnewname + '.npy'))
            cam_path.append(os.path.join(cam_dir, fnewname + '.npy'))
    return im_list, im_labels, roi_path, dest_path, cam_path
	
def load_data(im_dir, roi_dir, dest_dir, file_path, batch_size, isroi=False, cam_dir='/tmp/mytmp'):

    im_list, im_labels, roi_path, dest_path, cam_path = get_im_list(im_dir, roi_dir,
                                                                    dest_dir, file_path, cam_dir)

    height = width = imgsize
    MEAN_VALUE = None
    norm_value = [[0.,0.,0.],[1.,1.,1.]]

    def _read_function_tf(impath, label, rpath):
        im_f = tf.read_file(impath)
        oim = tf.image.decode_jpeg(im_f, channels=3)
        rim = tf.image.resize_images(oim, [height, width])
        if MEAN_VALUE is not None:
            # convert RGB to BGR
            rim = tf.cast(tf.reverse(rim, axis=[-1]), tf.float32)
            mean_image = tf.convert_to_tensor(
                np.tile(MEAN_VALUE, [height, width, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            print('mean norm')
        elif norm_value is not None:
            rim = rim / 255.0
            mean_image = tf.convert_to_tensor(
                np.tile(norm_value[0], [height, width, 1]), tf.float32)
            rim = tf.subtract(rim, mean_image)
            rim /= norm_value[1]
            print('standard norm')
        # rim = tf.image.resize_image_with_crop_or_pad(rim, height, width)
        # rois = tf.convert_to_tensor(roi_data(image_size, rsize, rstride))
        return rim, label, rpath

    def _read_function(impath, label, rpath, cpath):
        im = cv2.imread(impath)  # default BGR order
        rim = cv2.resize(im, (height, width))
        if MEAN_VALUE is not None:
            mean_image = np.tile(MEAN_VALUE, [height, width, 1])
            rim = rim - mean_image
            # print('mean norm')
        elif norm_value is not None:
            # convert BGR to RGB
            rim = rim[:, :, ::-1]
            rim = rim / 255.0
            mean_image = np.tile(norm_value[0], [height, width, 1])
            rim = rim - mean_image
            rim /= norm_value[1]
            # print('standard norm')
        return rim.astype(np.float32), label, rpath, cpath

    def _read_roi_function(impath, label, rpath, dpath):
        im = cv2.imread(impath)  # default BGR order
        rim = cv2.resize(im, (height, width))
        if MEAN_VALUE is not None:
            mean_image = np.tile(MEAN_VALUE, [height, width, 1])
            rim = rim - mean_image
            # print('mean norm')
        elif norm_value is not None:
            # convert BGR to RGB
            rim = rim[:, :, ::-1]
            rim = rim / 255.0
            mean_image = np.tile(norm_value[0], [height, width, 1])
            rim = rim - mean_image
            rim /= norm_value[1]
            # print('standard norm')
        rois = np.load(rpath)
        return rim.astype(np.float32), rois.astype(np.float32), label, dpath

    if isroi:
        dataset = tf.data.Dataset.from_tensor_slices((im_list, im_labels, roi_path, dest_path))
        map_func = lambda impath, label, rpath, dpath: tuple(tf.py_func(
            _read_roi_function, [impath, label, rpath, dpath], [tf.float32, tf.float32, tf.int32, tf.string]))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((im_list, im_labels, roi_path, cam_path))
        map_func = lambda impath, label, rpath, cpath: tuple(tf.py_func(
            _read_function, [impath, label, rpath, cpath], [tf.float32, tf.int32, tf.string, tf.string]))
    dataset = dataset.map(map_func)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    # not call iterator.get_next() in the loop, call next_element = iterator.get_next() once
    # outside the loop, and use next_element inside the loop
    next_element = iterator.get_next()

    return iterator, next_element

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
    #def __init__(self, args, myconfig):
        args=[]
        #myconfig = None
        self.args = args
        # Collect parameters:
        params = self.default_params()

        if myconfig is not None:
            params.update(myconfig)
        self.params = params

        # Collect argument things:
        self.train_dir = '/home/cll/fewshotlearning/GCN_feature/'
        #self.img_dir = '/home/cll/SUN397_224/'
        self.img_dir = '/home/0_public_data/MIT67/Images'
        self.roi_dir = self.train_dir + 'cluster_based/rois1shot'
        

        #self.train_data = json.load(open('sun_oneshot_train.json','r'))
        self.train_data = json.load(open('mit_oneshot_train.json','r'))

        #self.test_data = json.load(open('sun_oneshot_test.json','r'))
        self.test_data = json.load(open('mit_oneshot_test.json','r'))

        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.set_random_seed(params['random_seed'])

        self.placeholders = {}
        self.weights = {}
        self.ops = {}
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
        with tf.variable_scope("meta_model"):
            self.weights['meta1'] = MLP(2,1,[20,20],1.0)
            self.weights['meta2'] = MLP(2,1,[20,20],1.0)
        with tf.variable_scope("out_layer"):
            with tf.variable_scope("regression_gate"):
                self.weights['regression_node'] = MLP(self.params['out_size'], 5, [],
                                                      self.placeholders['out_layer_dropout_keep_prob'])
            with tf.variable_scope("regression"):
                    self.weights['regression_transform'] = MLP(self.params['out_size'], 5, [],
                                                               self.placeholders['out_layer_dropout_keep_prob'])
        self.get_grad()
        self.make_model()
        self.make_train_step()
        self.initialize_model()
        self.update_rois()

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

    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)


    def prepare_specific_graph_model(self):
        self.placeholders['support_x'] = tf.placeholder(tf.float32,[5,224,224,3],name='support_x')
        self.placeholders['support_roi'] = tf.placeholder(tf.float32,[5,self.params['n_cluster'],6],name='support_roi')
        self.placeholders['target_x'] = tf.placeholder(tf.float32,[40,224,224,3],name='target_x')
        self.placeholders['target_roi'] = tf.placeholder(tf.float32,[40,self.params['n_cluster'],6],name='target_roi')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, None)
        self.placeholders['support_adj'] = tf.placeholder(tf.float32,[5,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='support_adj')
        self.placeholders['target_adj'] = tf.placeholder(tf.float32,[40,self.params['n_cluster']+1,self.params['n_cluster']+1,self.params['edge_dims']],name='target_adj')
        
        self.placeholders['support_label'] = tf.placeholder(tf.int32, [5])
        self.placeholders['target_label'] = tf.placeholder(tf.int32, [40])
        #MIT67-5280 SUN397-9760
        self.placeholders['grad'] = tf.placeholder(tf.float32, [5, None, 2])
        #self.placeholders['grad'] = tf.placeholder(tf.float32, [5, 9760, 2])
        self.placeholders['grad1'] = tf.placeholder(tf.float32, [5, 5, 2])
        self.placeholders['support_v'] = tf.placeholder(tf.int32, [5])
        self.placeholders['target_v'] = tf.placeholder(tf.int32, [40])
        self.placeholders['is_training'] = tf.placeholder(tf.bool)
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')


    def gated_regression_v6(self, last_h, regression_node, regression_transform):
        # last_h: [b x v x h]
        last_h1 = tf.reshape(last_h[:, 1:, :], [-1, self.params['out_size']])
        node_out = regression_node(last_h1)
        print(node_out.shape)
        self.node_out = node_out
        last_h = last_h[:, 0, :]
        output = regression_transform(last_h)
        self.output = output
        return output
		
    def compute_keys(self, input):
        is_training = self.placeholders['is_training']
        with tf.variable_scope('keys', reuse=tf.AUTO_REUSE) as scope:
            x = conv(input, 3, 64, 1, pad=1, name='keys_1_conv', with_bias=True)
            x = BatchNorm(x, is_training, name='keys_1_bn')
            x = tf.nn.relu(x)
            print(x.shape)
            x = max_pool(x, 2, 2, padding='SAME')
            print(x.shape)

            x = conv(x, 3, 64, 1, pad=1, name='keys_2_conv', with_bias=True)
            x = BatchNorm(x, is_training, name='keys_2_bn')
            x = tf.nn.relu(x)
            print(x.shape)
            x = max_pool(x, 2, 2, padding='SAME')
            print(x.shape)

            x = conv(x, 3, 64, 1, pad=1, name='keys_3_conv', with_bias=True)
            x = BatchNorm(x, is_training, name='keys_3_bn')
            x = tf.nn.relu(x)
            x = max_pool(x, 2, 2, padding='SAME')
            print(x.shape)

            x = conv(x, 3, 64, 1, pad=1, name='keys_4_conv', with_bias=True)
            x = BatchNorm(x, is_training, name='keys_4_bn')
            x = tf.nn.relu(x)
            x = max_pool(x, 2, 2, padding='SAME')
            print(x.shape)
        
        return tf.contrib.layers.flatten(x)        
        
    def run_cnn(self, X, is_training=False):
        x = conv(X, 3, 64, 1, pad=1, name='clf1_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf1_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')

        x = conv(x, 3, 64, 1, pad=1, name='clf2_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf2_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')

        x = conv(x, 3, 64, 1, pad=1, name='clf3_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf3_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')

        layer4 = conv(x, 3, 64, 1, pad=1, name='clf4_conv', with_bias=True)
        layer4 = BatchNorm(layer4, is_training, name='clf4_bn')
        layer4 = tf.nn.leaky_relu(layer4)
        layer4 = max_pool(layer4, 2, 2, padding='SAME')
        return layer4
    
    
    def extract_salient_region_scda_adapt(self,sess, tfcam, fdict, batch_input):
        batch_im, batch_label, batch_rpath, batch_cpath = batch_input
        ocam = sess.run([tfcam], feed_dict={fdict['x']: batch_im})
        ocam = np.squeeze(ocam)
        ocam = np.transpose(ocam, [2, 0, 1])
        ocam_valid_check = ocam.max(axis=(1, 2))
        thre = ocam_valid_check.mean()
        ocam_vind = np.where(ocam_valid_check > thre)
        ocam_valid = ocam[ocam_vind[0]]
        boxes = []
        for ocv in ocam_valid:
            oim = cv2.resize(ocv, (imgsize, imgsize))
            box = findSalientRegion(oim, thre)
            if box.size:
                boxes.append(box)
        boxes = np.concatenate(boxes, axis=0)
        #print(ocam_valid.shape, boxes.shape)
        clustering = AgglomerativeClustering(n_clusters=min([self.params['n_cluster'],boxes.shape[0]]))
        clustering.fit(boxes)
        boxes_dedup = []
        for i in range(min([self.params['n_cluster'],boxes.shape[0]])):
            ind = np.where(clustering.labels_ == i)[0]
            boxes_pick = boxes[ind]
            boxes_dedup.append(boxes_pick.mean(axis=0))
        
        if boxes.shape[0]<self.params['n_cluster']:
            for i in range(self.params['n_cluster']-boxes.shape[0]):
                boxes_dedup.append([224.,224.,0.,0.,0.,0.])
        boxes_dedup = np.array(boxes_dedup)
       
        np.save(batch_rpath[0], boxes_dedup)
    
    def update_rois(self):
        
        data = self.train_data.copy()
        data.update(self.test_data)
        
        batch_size = 1
        roi_dir = prepare_dir(self.roi_dir)
        cam_dir = prepare_dir('./cams')
        print(roi_dir)
        train_file = './TrainImages.label'
        #train_file = './SUNTrainImages.txt'
        test_file = './TestImages.label'
        #test_file = './SUNTestImages.txt'
        train_iter, train_data = load_data(self.img_dir, roi_dir, '', train_file, batch_size, False, cam_dir=cam_dir)
        test_iter, test_data = load_data(self.img_dir, roi_dir, '', test_file, batch_size, False, cam_dir=cam_dir)
        x = tf.placeholder(tf.float32, [None, imgsize, imgsize, 3])
        cate_id = tf.placeholder(tf.int32)
        fdict = {'x': x, 'cate_id': cate_id}
        with tf.variable_scope("graph_model", reuse = tf.AUTO_REUSE):
            model_out = self.run_cnn(x)
        # Configuration of GPU usage
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # extract salient regions
            sess.run(train_iter.initializer)
            count = 0
            vstart_time = time.time()
            while True:
                try:
                    count += 1
                    batch_input = sess.run(train_data)
                    self.extract_salient_region_scda_adapt(sess, model_out, fdict, batch_input)
                    if count % 1000 == 0:
                        print('{} images are done, {:.4f}s per image'.format(
                            count, (time.time()-vstart_time) / count))
                except tf.errors.OutOfRangeError:
                    break
            # extract test feature
            sess.run(test_iter.initializer)
            count = 0
            vstart_time = time.time()
            while True:
                try:
                    count += 1
                    batch_input = sess.run(test_data)
                    self.extract_salient_region_scda_adapt(sess, model_out, fdict, batch_input)
                    if count % 1000 == 0:
                        print('{} images are done, {:.4f}s per image'.format(
                            count, (time.time() - vstart_time) / count))
                except tf.errors.OutOfRangeError:
                    break


    def get_feature(self, X, rois, is_training=False):
        output = {}
        print('############################')
        # Layer 1 in (56x56)
        x = conv(X, 3, 64, 1, pad=1, name='clf1_conv', with_bias=True)
        output['conv1'] = x
        x = BatchNorm(x, is_training, name='clf1_bn')
        x = tf.nn.leaky_relu(x)
        output['bn1'] = x
        print(x.shape)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)
        #output['layer1'] = x

        x = conv(x, 3, 64, 1, pad=1, name='clf2_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf2_bn')
        x = tf.nn.leaky_relu(x)
        print(x.shape)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)
        #output['layer2'] = x

        x = conv(x, 3, 64, 1, pad=1, name='clf3_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf3_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')
        print(x.shape)
        #output['layer3'] = x

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
        
        h = tf.concat([output['gpool5'][:, None, :], output['roi_conv']], axis=1)  # [b, v, h]

        return h
        
    def compute_final_node_representations_v16(self, node, edge, grad, name):
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
        e = edge  # [b, v, v, e_dim]
        e = tf.reshape(e, [-1, e_dim])
        self.weights['proj_h_sim1'] = fc(e, 1, 'proj_e', with_bias=False, relu=False, reuse=tf.AUTO_REUSE)
        if grad is not None:
            self.weights['proj_h_sim1'] += grad['proj_h_sim1']
        # only diag mask
        a = tf.reshape(self.weights['proj_h_sim1'], [-1, v, v])
        a -= inf_mask[None, :, :]
        a = tf.nn.softmax(a, axis=-1)
        a = a + diag_I[None, :, :]
        self.ops['adjm'] = a

        d = tf.matrix_diag(1 / tf.reduce_sum(a, axis=-1))
        sa = tf.matmul(d, a)
        

        for i in range(self.params['num_timesteps']):
            sah = tf.matmul(sa, h)
            sah = tf.reshape(sah, [-1, in_dim])
            self.weights['gcn_{}_sim'.format(i)] = fc(sah, h_dim, name + 'gcn_{}_sim'.format(i), with_bias=False, relu=False, reuse=tf.AUTO_REUSE)
            if grad is not None:
                self.weights['gcn_{}_sim'.format(i)] += grad['gcn_{}_sim'.format(i)]
            self.weights['gcn_{}_sim'.format(i)] = tf.nn.relu(self.weights['gcn_{}_sim'.format(i)])
            smh = tf.reshape(self.weights['gcn_{}_sim'.format(i)], [-1, v, h_dim])
            h = smh
            #h = tf.maximum(ngh, smh)
            #h = tf.nn.l2_normalize(h, axis=-1)

        h += init_h
        return h

    def get_grad(self):
        with tf.variable_scope('graph_model', reuse=tf.AUTO_REUSE):
            support_x = self.get_feature(self.placeholders['support_x'],self.placeholders['support_roi'],self.placeholders['is_training'])
        support_y = self.placeholders['support_label']
        support_n = support_x.shape[0]
        tv = self.placeholders['num_vertices'] - 1
        grad=[]
        grad1=[]
        sloss = []
        edge = self.placeholders['support_adj']
        print(support_x.shape)
        for i in range(support_n):
            support_i = support_x[i][None,:,:]
            #print(support_i.shape)
            support_representation = self.compute_final_node_representations_v16(support_i, edge[i], None, name = 'base')
            #print(support_representation.get_shape().as_list())
            pred = self.gated_regression_v6(support_representation, self.weights['regression_node'], self.weights['regression_transform'])
            #print(pred.get_shape().as_list())
            support_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(support_y[i],[-1]), logits=pred)
            sloss.append(support_loss)
            
            #tv = tf.minimum(self.placeholders['support_v'][i], self.params['n_cluster'])
            node_labels = tf.tile(tf.reshape(support_y[i],[1]),[tv])
            node_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node_labels,logits=self.node_out[:tv,:])
            
            loss = support_loss + node_loss * self.params['node_lambda']
            gd = []
            gd_size = []
            #gd.append(tf.reshape(tf.gradients(loss, self.weights['proj_h_d']), [-1, 1]))
            #gd_size.append(gd[-1].get_shape().as_list()[0])
            #gd.append(tf.reshape(tf.gradients(loss, self.weights['gcn_0_normal']), [-1, 1]))
            #gd_size.append(gd[-1].get_shape().as_list()[0])
            gd.append(tf.reshape(tf.gradients(loss, self.weights['proj_h_sim1']), [-1, 1]))
            gd_size.append(gd[-1].get_shape().as_list()[0])
            gd.append(tf.reshape(tf.gradients(loss, self.weights['gcn_0_sim']), [-1, 1]))
            gd_size.append(gd[-1].get_shape().as_list()[0])
            gd1 = []
            gd1_size = []
            gd1.append(tf.reshape(tf.gradients(loss, pred), [-1, 1]))
            gd1_size.append(gd1[-1].get_shape().as_list()[0])

            grad.append(gd)
            grad1.append(gd1)

        self.ops['support_grad'] = {
        'loss': tf.reduce_mean(sloss),
        'grad1': grad,
        'grad2': grad1,
        }



    def make_model(self):
        
        grad = self.placeholders['grad']
        grad1 = self.placeholders['grad1']
        meta1=[]
        meta2=[]
        for i in range(5):
            meta1.append(tf.reshape(self.weights['meta1'](grad[i]), [1,-1]))
            meta2.append(tf.reshape(self.weights['meta2'](grad1[i]), [1,-1]))
        meta1 = tf.concat(meta1,axis=0)
        meta2 = tf.concat(meta2,axis=0)
        print(meta1.get_shape().as_list())
        with tf.variable_scope('graph_model', reuse=tf.AUTO_REUSE):
            target_x = self.get_feature(self.placeholders['target_x'],self.placeholders['target_roi'],self.placeholders['is_training'])
        target_y = self.placeholders['target_label']
        print(target_y[0].get_shape().as_list())
        support_keys = self.compute_keys(self.placeholders['support_x'])
        target_keys = self.compute_keys(self.placeholders['target_x'])
        tv = self.placeholders['num_vertices'] - 1
        target_n = target_x.shape[0]
        node_loss = []
        loss = []
        acc = []
        edge = self.placeholders['target_adj']
        for i in range(target_n):
            target_i = target_x[i]
            target_k = target_keys[i]

            sc = tf.nn.softmax(cosine_d(tf.reshape(target_k,[1,-1]),tf.reshape(support_keys,[5,-1])), axis=1)
            target_meta1 = tf.matmul(sc, meta1)
            target_meta2 = tf.matmul(sc, meta2)
            num = self.placeholders['num_vertices']*self.placeholders['num_vertices']
            csn = {
                'proj_h_sim1': tf.reshape(target_meta1[0][:num],[num,1]),
                'gcn_0_sim': tf.reshape(target_meta1[0][num:],[self.placeholders['num_vertices'],self.params['hidden_size']])
            }
            target_representation = self.compute_final_node_representations_v16(target_i[None,:,:], edge[i], csn, name = 'base')
            pred = self.gated_regression_v6(target_representation, self.weights['regression_node'], self.weights['regression_transform'])
            pred += target_meta2
            target_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(target_y[i],[-1]), logits=pred)
            node_labels = tf.tile(tf.reshape(target_y[i],[1]),[tv])
            node_loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node_labels,logits=self.node_out[:tv,:]))
            loss.append(target_loss)
            acc.append(tf.equal(tf.argmax(pred, 1), tf.cast(target_y[i], tf.int64)))
        node_loss = tf.reduce_mean(tf.concat(node_loss,0))
        loss = tf.reduce_mean(loss)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        self.ops['accuracy'] = acc
        self.ops['loss'] = loss + node_loss * self.params['node_lambda']


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
                selected_samples = np.random.choice(len(dirs), 1+8, False)
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
                self.placeholders['num_vertices']: num_node + 1,
            }

            if is_training:
                batch_data[self.placeholders['is_training']] = True
                fetch_list = [self.ops['loss'], accuracy_op, self.ops['train_step']]

            else:
                batch_data[self.placeholders['is_training']] = False
                fetch_list = [self.ops['loss'], accuracy_op]
            support_grads = self.sess.run(self.ops['support_grad'], feed_dict = batch_data)
            grad=[]
            grad1=[]
            size=[]
            for i in range(len(support_grads['grad1'][0])):
                size.append(support_grads['grad1'][0][i].shape[0])
            #print size
            for i in range(5):
                metain = np.concatenate(support_grads['grad1'][i], axis = 0)
                metain = logAndSign(metain, k=7)
                metain = np.array(metain)
                grad.append(metain[None,:,:])
                metain = np.concatenate(support_grads['grad2'][i], axis = 0)
                metain = logAndSign(metain, k=7)
                metain = np.array(metain)
                grad1.append(metain[None,:,:])
            grad=np.concatenate(grad,axis=0)
            grad1=np.concatenate(grad1,axis=0)
            #print(grad.shape)
            #print(grad1.shape)
            batch_data[self.placeholders['grad']] = grad
            batch_data[self.placeholders['grad1']] = grad1
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
