# -*- coding: utf-8 -*-
import tensorflow as tf
#from docopt import docopt
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
from sklearn.cluster import k_means, AgglomerativeClustering
from tensorflow.python.ops import array_ops
from utils import prepare_dir, roi_pool_layer, global_pool, MLP, MLP_BN, fc
from ResNet import max_pool, Relu, BatchNorm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from PIL import Image
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
imgsize = 224
PiLImageResize = lambda x: x.resize((imgsize,imgsize))

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
        self.make_model()
        self.make_train_step()
        self.initialize_model()
        self.update_rois()




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

        
        
    def run_cnn(self, X, is_training=False):
        x = conv(X, 3, 64, 1, pad=1, name='clf1_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf1_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')
        # Layer 2 in (56x56)
        x = conv(x, 3, 64, 1, pad=1, name='clf2_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf2_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')
        # Layer 3 in (56x56)
        x = conv(x, 3, 64, 1, pad=1, name='clf3_conv', with_bias=True)
        x = BatchNorm(x, is_training, name='clf3_bn')
        x = tf.nn.leaky_relu(x)
        x = max_pool(x, 2, 2, padding='SAME')
        # Layer 4 in (28x28)
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


    def get_feature(self, X, rois, edge, is_training=False):
        output = {}
        print('############################')
        layer4 = self.run_cnn(X,is_training)
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
        # for similarity graph
        ph_dim = int(in_dim / 2)
        # ph_dim = in_dim
        ph = fc(tf.reshape(h, [-1, in_dim]), ph_dim, 'proj_h_sim1', with_bias=False, relu=False, reuse=tf.AUTO_REUSE)
        ph = tf.reshape(ph, [-1, v, ph_dim])
        ph = tf.nn.l2_normalize(ph, dim=-1)
        hv = tf.tile(ph[:, :, None, :], [1, 1, v, 1])
        hw = tf.tile(ph[:, None, :, :], [1, v, 1, 1])
        att_s = tf.reduce_sum(hv * hw, axis=-1)
        att_s -= inf_mask[None, :, :]
        sa = tf.nn.softmax(att_s)
        sa = sa + diag_I[None, :, :]
        self.ops['adjm'] = sa
        sd = tf.matrix_diag(1 / tf.reduce_sum(sa, axis=-1))
        sa = tf.matmul(sd, sa)
        
        for i in range(self.params['num_timesteps']):
            # for similarity graph
            sah = tf.matmul(sa, h)
            sah = tf.reshape(sah, [-1, in_dim])
            smh = fc(sah, h_dim, 'gcn_{}_sim'.format(i), with_bias=False, relu=True, reuse=tf.AUTO_REUSE)
            smh = tf.reshape(smh, [-1, v, h_dim])
            #h = tf.maximum(ngh, smh)
            h = smh
            #h = tf.nn.l2_normalize(h, axis=-1)
        h += init_h
        return h


    def make_model(self):
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')
        
        with tf.variable_scope("graph_model", reuse = tf.AUTO_REUSE):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            support_final_node_representations = self.get_feature(self.placeholders['support_x'],self.placeholders['support_roi'],self.placeholders['support_adj'],self.placeholders['is_training'])
            #[5,v,4096]
            target_final_node_representations = self.get_feature(self.placeholders['target_x'],self.placeholders['target_roi'],self.placeholders['target_adj'],self.placeholders['is_training'])
            #[75,v,4096]
        with tf.variable_scope("out_layer", reuse = tf.AUTO_REUSE):
            
            with tf.variable_scope("regression_gate"):
                self.weights['regression_node'] = MLP(self.params['out_size'], 5, [],
                                                      self.placeholders['out_layer_dropout_keep_prob'])
        
        node_loss = tf.constant(0.0)
        
        print('Node supervision')
        tv = self.placeholders['num_vertices']-1
        #sv = self.placeholders['support_v']
        v = self.placeholders['num_vertices']
        #qv = self.placeholders['target_v']
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
        
        support_final_node_representations = tf.reshape(support_final_node_representations[:,0,:],[5,self.params['out_size']])
        target_final_node_representations = tf.reshape(target_final_node_representations[:,0,:],[50,self.params['out_size']])
        
        similarities = cosine_d(target_final_node_representations,support_final_node_representations)
        
        similarities = tf.reshape(similarities, [50, 5])         
        support_set_labels = tf.one_hot(self.placeholders['support_label'], 5)
        preds = tf.squeeze(tf.matmul(tf.nn.softmax(similarities), support_set_labels))
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.placeholders['target_label'], tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        targets = tf.one_hot(self.placeholders['target_label'], 5)
        print("preds ",preds.get_shape().as_list())   
        print("targets ",targets.get_shape().as_list())   
        mean_square_error_loss = tf.reduce_mean(tf.square((preds-1)*targets + preds*(1-targets)))
        self.ops['accuracy'] = accuracy
        self.ops['loss'] = mean_square_error_loss + node_loss


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
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                batch_data[self.placeholders['is_training']] = True
                fetch_list = [self.ops['loss'], accuracy_op, self.ops['train_step']]

            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                batch_data[self.placeholders['is_training']] = False
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
