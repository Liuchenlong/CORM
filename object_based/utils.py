# -*- coding: utf-8 -*-
"""
Created on 4/27/18

Description: 

@author: Gongwei Chen
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from tensorflow.python.ops import array_ops

SMALL_NUMBER = 1e-7


def prepare_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    np.random.seed(1)
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


def dropout_node(x, keep_prob, name=None):

    with tf.name_scope(name, "dropout_node", [x]) as name:
        x = tf.convert_to_tensor(x, name='x')
        keep_prob = tf.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        # keep_prob.get_shape().assert_is_compatible_with(tf.tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tf.contrib.util.constant_value(keep_prob) == 1:
            return x

        # only dropout [0, 1] dimensions.
        bshape = array_ops.shape(x)[:-1]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(
            bshape, seed=0, dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        ret = x * binary_tensor[:, :, None]
        return ret


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob, bias=True):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.with_bias = bias
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        if self.with_bias:
            biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                      for (i, s) in enumerate(weight_sizes)]
        else:
            biases = [None for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        # return tf.truncated_normal(shape)  # better for gru output and mean all nodes
        # return tf.random_uniform(shape)
        return glorot_init(shape)
        # better for relu output, (just l2 norm in and mean out)
        # glorot normal init
        # return np.sqrt(2.0 / (shape[-2] + shape[-1])) * np.random.rand(*shape).astype(np.float32)
        # return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (1 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for i, (W, b) in enumerate(zip(self.params["weights"], self.params["biases"])):
            # if i > 0:
            acts = tf.nn.dropout(acts, self.dropout_keep_prob)
            # This implement is unstable
            # hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            if b is not None:
                hid = tf.matmul(acts, W) + b
            else:
                hid = tf.matmul(acts, W)
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


class MLP_v2(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        # self.name = name
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        # with tf.variable_scope(self.name):
        weights = [tf.Variable(np.zeros(s).astype(np.float32), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        # return tf.truncated_normal(shape)  # better for gru output and mean all nodes
        return glorot_init(shape)
        # better for relu output, (just l2 norm in and mean out)
        # glorot normal init
        # return np.sqrt(2.0 / (shape[-2] + shape[-1])) * np.random.rand(*shape).astype(np.float32)
        # return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (1 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for i, (W, b) in enumerate(zip(self.params["weights"], self.params["biases"])):
            # if i > 0:
            acts = tf.nn.dropout(acts, self.dropout_keep_prob)
            # This implement is unstable
            # hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            hid = tf.matmul(acts, W) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


class MLP_BN(object):
    def __init__(self, in_size, out_size, hid_sizes,
                 dropout_keep_prob, is_training, name):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.name = name
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        with tf.variable_scope(self.name):
            weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                       for (i, s) in enumerate(weight_sizes)]
            biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                      for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        # return tf.truncated_normal(shape)  # better for gru output and mean all nodes
        # return glorot_init(shape)
        # better for relu output, (just l2 norm in and mean out)
        # glorot normal init
        return np.sqrt(2.0 / (shape[-2] + shape[-1])) * np.random.rand(*shape).astype(np.float32)
        # return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (1 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for i, (W, b) in enumerate(zip(self.params["weights"], self.params["biases"])):
            with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
                acts = BatchNorm(acts, is_training=self.is_training,
                             name='{}_BN_{}'.format(self.name, i))
            if i > 0:
                acts = tf.nn.dropout(acts, self.dropout_keep_prob)
            # This implement is unstable
            # hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            [batch,h,w,c] = acts.get_shape().as_list()
            acts=tf.reshape(acts, [-1,c])
            hid = tf.matmul(acts, W) + b
            hid = tf.reshape(hid, [batch,h,w,-1])
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


def classifier(train_feat, train_label, test_feat, test_label):
    # para = {'C': np.array([10.0, 100.0, 0.1, 1.0]).tolist()}
    # svc = LinearSVC(random_state=0, max_iter=10000)
    # clf = GridSearchCV(svc, para, n_jobs=4, cv=10, scoring=mr_f)
    clf = LinearSVC(C=1)

    # print('------Starting train Linear SVM model-----')
    # startt = time.time()
    # X's size is [num_samples, num_features], y's size is [num_samples]
    clf.fit(train_feat, train_label)
    # acc = clf.score(train_feat, train_label)
    # print('------Training is done in {:.3f}s-----\n'.format(time.time() - startt))
    # print('train accuracy is {:.4f}'.format(acc))

    # print('-----Grid Search best result-----')
    # print(clf.best_params_)
    # print(clf.best_score_)

    print('\n-----Starting test model-----')
    acc = clf.score(test_feat, test_label)
    print('test accuracy is {:.4f}'.format(acc))

    return acc


def softmax(X, axis=-1):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    axis: axis to compute values along. Default is the
        last axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.amax(y, axis=axis, keepdims=True)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.sum(y, axis=axis, keepdims=True)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def relu_np(x):
    return np.maximum(x, 0)


def L2_norm_np(x, axis=-1):
    eps = 1e-12
    x = x / np.sqrt(np.maximum(np.sum(x**2, axis=axis, keepdims=True), eps))
    return x


def L1_norm_tf(x, axis=-1):
    x = x - tf.reduce_min(x, axis=axis, keepdims=True)
    x = x / tf.reduce_sum(x, axis=axis, keepdims=True)
    return x


def L1_norm_np(x, axis=-1):
    x = x - np.min(x, axis=axis, keepdims=True)
    x = x / np.sum(x, axis=axis, keepdims=True)
    return x


def euclid_dist(A, B, eps=1e-12):
    # A, B need to be 2-dim, [num_samples, feat_dim]
    M = np.dot(A, B.T)
    H = np.square(A).sum(axis=1, keepdims=True)
    K = np.square(B).sum(axis=1, keepdims=True)
    D = -2*M+H+K.T
    return D, np.sqrt(np.maximum(D, eps))


def cosine_dist(A, B):
    # A, B need to be 2-dim, [num_samples, feat_dim]
    # D's value is in range [0, 2], higher value, bigger theta
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[-1] == B.shape[-1]
    A = L2_norm_np(A)
    B = L2_norm_np(B)
    An = A.shape[0]
    Bn = B.shape[0]
    dA = np.tile(A[:, None, :], [1, Bn, 1])
    dB = np.tile(B[None, :, :], [An, 1, 1])
    D = np.sum(dA * dB, axis=-1)
    return 1 - D

def L2_norm_tf(x, axis=-1):
        eps = 1e-12
        x = x / tf.sqrt(tf.maximum(tf.reduce_sum(x**2, axis=axis, keepdims=True), eps))
        return x

def cosine_d(A, B):
        # A, B need to be 2-dim, [num_samples, feat_dim]
        # D's value is in range [0, 2], higher value, bigger theta
        #assert A.ndim == 2 and B.ndim == 2
        #assert A.shape[-1] == B.shape[-1]
        A = L2_norm_tf(A)
        B = L2_norm_tf(B)
        An = A.shape[0]
        Bn = B.shape[0]
        dA = tf.stack([A]*Bn, axis=1)#np.tile(A[:, None, :], [1, Bn, 1])
        dB = tf.stack([B]*An)#np.tile(B[None, :, :], [An, 1, 1])
        D = tf.reduce_sum(dA * dB, axis=-1)
        return D


def kl_tf(p, q, v):
    # Symmetrised Kullback–Leibler divergence
    # p, q size, [batch_size, num_sample, feat_dim]
    p += SMALL_NUMBER
    q += SMALL_NUMBER
    p /= tf.reduce_sum(p, axis=-1, keepdims=True)
    q /= tf.reduce_sum(q, axis=-1, keepdims=True)
    # v = p.shape[1]
    p = tf.tile(p[:, :, None, :], [1, 1, v, 1])
    q = tf.tile(q[:, None, :, :], [1, v, 1, 1])
    Dpq = -tf.reduce_sum((p*tf.log(p/q)), axis=-1)
    Dqp = -tf.reduce_sum((q*tf.log(q/p)), axis=-1)
    return Dpq + Dqp


def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    # position max, [num_rois, num_rois, 4]
    nr, nc, nd = position_mat.shape
    assert nr == nc
    assert nd == 4
    feat_range = np.arange(0, feat_dim // 8)
    dim_mat = np.power([wave_length], (8. / feat_dim)*feat_range)
    dim_mat = np.reshape(dim_mat, [1, 1, 1, -1])
    position_mat = np.expand_dims(100*position_mat, axis=3)
    div_mat = position_mat / dim_mat
    sin_mat = np.sin(div_mat)
    cos_mat = np.cos(div_mat)
    # embedding, [num_rois, num_rois, 4, feat_dim/4]
    embedding = np.concatenate([sin_mat, cos_mat], axis=3)
    # embedding, [num_rois, num_rois, feat_dim]
    embedding = np.reshape(embedding, [nr, nc, -1])
    return embedding


def extract_position_embedding_coords(position_mat, feat_dim, wave_length=1000):
    # position max, [num_rois, num_rois, 4]
    nr, nc, nd = position_mat.shape
    assert nr == nc
    assert nd == 2
    feat_range = np.arange(0, feat_dim // 4)
    dim_mat = np.power([wave_length], (4. / feat_dim)*feat_range)
    dim_mat = np.reshape(dim_mat, [1, 1, 1, -1])
    position_mat = np.expand_dims(100*position_mat, axis=3)
    div_mat = position_mat / dim_mat
    sin_mat = np.sin(div_mat)
    cos_mat = np.cos(div_mat)
    # embedding, [num_rois, num_rois, 2, feat_dim/2]
    embedding = np.concatenate([sin_mat, cos_mat], axis=3)
    # embedding, [num_rois, num_rois, feat_dim]
    embedding = np.reshape(embedding, [nr, nc, -1])
    return embedding


"""
Predefine all necessary layer for CNN
"""


def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            # initializer = tf.contrib.layers.xavier_initializer(seed=1)
            initializer = tf.glorot_uniform_initializer()
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def roi_pool_layer_wglobal(bottom, rois, POOLING_SIZE, batch_size, name):

    with tf.variable_scope(name):

        # # using a single image as input
        # batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # # Get the normalized coordinates of bounding boxes
        # bottom_shape = tf.shape(bottom)
        # height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        # width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
        # x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        # y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        # x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        # y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        # # Won't be back-propagated to rois anyway, but to save time
        # bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        # pooled = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
        #                                   [POOLING_SIZE, POOLING_SIZE])

        # bottom size is [batch_size, H, W, C]
        # rois size is [batch_size, num_bbox, (width, height, x_min, y_min, x_max, y_max)]
        num_bbox = rois.shape.as_list()[1]
        batch_ids = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, num_bbox]), [-1])
        print('batch_ids', batch_size)
        print('rois', num_bbox)
        tp_rois = tf.reshape(rois, [batch_size*num_bbox, -1])
        x1 = tf.slice(tp_rois, [0, 2], [-1, 1], name="x1") / tf.slice(tp_rois, [0, 0], [-1, 1], name="width")
        y1 = tf.slice(tp_rois, [0, 3], [-1, 1], name="y1") / tf.slice(tp_rois, [0, 1], [-1, 1], name="height")
        x2 = tf.slice(tp_rois, [0, 4], [-1, 1], name="x2") / tf.slice(tp_rois, [0, 0], [-1, 1], name="width")
        y2 = tf.slice(tp_rois, [0, 5], [-1, 1], name="y2") / tf.slice(tp_rois, [0, 1], [-1, 1], name="height")
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pooled = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                          [POOLING_SIZE, POOLING_SIZE])
        # print(pooled.shape)
        # output pooled size [batch_size, num_bbox, POOLING_SIZE*POOLING_SIZE*Channels]
        pooled = tf.reshape(pooled, [batch_size, num_bbox, POOLING_SIZE, POOLING_SIZE, -1])
        pooled = tf.concat([bottom[:, None, :, :, :], pooled], axis=1)
        pooled = tf.reshape(pooled, [batch_size*(num_bbox+1), POOLING_SIZE, POOLING_SIZE, -1])
        print(pooled.shape)

    return pooled


def roi_pool_layer(bottom, rois, POOLING_SIZE, batch_size, name):

    with tf.variable_scope(name):

        # # using a single image as input
        # batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # # Get the normalized coordinates of bounding boxes
        # bottom_shape = tf.shape(bottom)
        # height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        # width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
        # x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        # y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        # x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        # y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        # # Won't be back-propagated to rois anyway, but to save time
        # bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        # pooled = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
        #                                   [POOLING_SIZE, POOLING_SIZE])

        # bottom size is [batch_size, H, W, C]
        # rois size is [batch_size, num_bbox, (width, height, x_min, y_min, x_max, y_max)]
        num_bbox = rois.shape.as_list()[1]
        batch_ids = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, num_bbox]), [-1])
        print('batch_ids', batch_size)
        print('rois', num_bbox)
        tp_rois = tf.reshape(rois, [batch_size*num_bbox, -1])
        x1 = tf.slice(tp_rois, [0, 2], [-1, 1], name="x1") / tf.slice(tp_rois, [0, 0], [-1, 1], name="width")
        y1 = tf.slice(tp_rois, [0, 3], [-1, 1], name="y1") / tf.slice(tp_rois, [0, 1], [-1, 1], name="height")
        x2 = tf.slice(tp_rois, [0, 4], [-1, 1], name="x2") / tf.slice(tp_rois, [0, 0], [-1, 1], name="width")
        y2 = tf.slice(tp_rois, [0, 5], [-1, 1], name="y2") / tf.slice(tp_rois, [0, 1], [-1, 1], name="height")
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pooled = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                          [POOLING_SIZE, POOLING_SIZE])
        # output pooled size [batch_size, num_bbox, POOLING_SIZE*POOLING_SIZE*Channels]
        pooled = tf.reshape(pooled, [batch_size*num_bbox, POOLING_SIZE, POOLING_SIZE, -1])

    return pooled


def roi_pool_layer_one(bottom, rois, POOLING_SIZE, name):

    with tf.variable_scope(name):

        # using a single image as input
        num_boxes = tf.shape(rois)[0]
        batch_ids = tf.tile([0], [num_boxes])
        # Get the normalized coordinates of bounding boxes
        x1 = tf.slice(rois, [0, 2], [-1, 1], name="x1") / tf.slice(rois, [0, 0], [-1, 1], name="width")
        y1 = tf.slice(rois, [0, 3], [-1, 1], name="y1") / tf.slice(rois, [0, 1], [-1, 1], name="height")
        x2 = tf.slice(rois, [0, 4], [-1, 1], name="x2") / tf.slice(rois, [0, 0], [-1, 1], name="width")
        y2 = tf.slice(rois, [0, 5], [-1, 1], name="y2") / tf.slice(rois, [0, 1], [-1, 1], name="height")
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = POOLING_SIZE * 2
        pooled = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                          [pre_pool_size, pre_pool_size])

    return max_pool(pooled, 2, 2, name='roi_after_mpool')


def conv(x, kernel_size, num_kernels, stride_size, name, pad=0,
         with_bias=False, reuse=tf.AUTO_REUSE, padding='VALID'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    # print(x.get_shape())

    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_size, stride_size, 1],
                                         padding=padding)
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_size, kernel_size,
                                               input_channels, num_kernels], 1e-1)
        biases = _variable_on_cpu('biases', [num_kernels], 0.0)

        # Apply convolution function
        conv = convolve(x, weights)

        # Add biases
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        #relu = tf.nn.leaky_relu(bias, name=scope.name)
        return bias


def fc(x, num_out, name, with_bias=True, relu=True, reuse=False):
    num_in = x.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):

        # Create tf variable for the weights and biases
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)

        # Matrix multiply weights and inputs
        act = tf.matmul(x, weights)

        if with_bias:
            # add bias
            biases = _variable_on_cpu('biases', [num_out], 0.0)
            act = tf.add(act, biases)

        if relu:
            # Apply ReLu non linearity
            act = tf.nn.relu(act)
        return act


def fc_LeakyReLU(x, num_out, name, with_bias=True, lkrelu=True, reuse=False):
    num_in = x.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):

        # Create tf variable for the weights and biases
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)

        # Matrix multiply weights and inputs
        act = tf.matmul(x, weights)

        if with_bias:
            # add bias
            biases = _variable_on_cpu('biases', [num_out], 0.0)
            act = tf.add(act, biases)

        if lkrelu:
            # Apply ReLu non linearity
            # act = tf.nn.relu(act)
            act = tf.nn.leaky_relu(act)
        return act


def max_pool(x, kernel_size, stride_size, pad=0, padding='VALID'):
    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding)


def global_pool(x, name, ptype='avg'):
    if ptype == 'avg':
        ssize = ksize = x.shape[1]
        return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                              strides=[1, ssize, ssize, 1],
                              padding='VALID', name=name)
    elif ptype == 'max':
        ssize = ksize = x.shape[1]
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                              strides=[1, ssize, ssize, 1],
                              padding='VALID', name=name)
    else:
        raise NotImplementedError


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def BatchNorm(x, is_training, name):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, scale=True,
                                        is_training=is_training, fused=True,
                                        zero_debias_moving_mean=False, scope=name)

def LayerNorm(x, name, center=False, scale=False, trainable=False):
    return tf.contrib.layers.layer_norm(x, center=center, scale=scale, trainable=trainable, scope=name)


def UnitNorm(x, axis, name):
    return tf.contrib.layers.unit_norm(x, dim=axis, scope=name)

def Relu(x):
    return tf.nn.relu(x)

