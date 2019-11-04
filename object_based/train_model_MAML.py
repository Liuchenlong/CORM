import numpy as np
import tqdm
from storage import *
import os
import time
from A1_B2_C3_1shot import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



gpu = 0
n_epoch   = 300   # number of epochs
batch_size = 1  # minibatch size
n_outputs = 5
n_outputs_test = 5
n_eposide = 400
n_eposide_tr = 400
nb_samples_per_class = 1
nb_samples_per_class_test = 15

print "batch_size", batch_size
print "GPU", gpu
print "n_classes (train):", n_outputs
print "n_classes (test):", n_outputs_test
print "nb_samples_per_class:", nb_samples_per_class
print "nb_samples_per_class_test:", nb_samples_per_class_test

print "train n_eposide:", n_eposide_tr
print "test n_eposide:", n_eposide

myconfig = {
	'learning_rate': 0.001,
	'num_timesteps': 1,
	'hidden_size': 128,
	'in_size': 64,
	'out_size': 128,
	'out_layer_dropout_keep_prob': 0.5,
	'random_seed': 1,
	'node_lambda': 1.0,
	'n_cluster': 30,
	'edge_dims': 10,
}

experimentname = 'A1_B2_C3_mit'

for test_epoch in range(1):
	model = GCNMetaModel(myconfig)
	log_to_save = []
	total_time_start = time.time()
	save_statistics(experimentname,['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'best_val_acc'])
	(best_val_acc, best_val_acc_epoch) = (0.0, 0)
	ls=0.0
	ac=0.0
	for epoch in range(100000):
		result1, result2, acc1, acc2 = model.run_epoch(is_training = True)
		ls = ls + result2
		ac = ac + acc2
		if (epoch != 0) and epoch % 100 == 0:
			print("=={} {}".format(experimentname,epoch))
			#print("\r\x1b[K %.5f | %.5f | %.5f | %.5f | " % (result1, result2, acc1, acc2))
			print("\r\x1b[K | %.5f | %.5f | " % (ls,ac))
			ls = 0.
			ac = 0.
		if (epoch != 0) and epoch % 1000 == 0:
			means,stds,ci95 = model.run_epoch(is_training = False)
			print((means,stds,ci95))
			save_statistics(experimentname,[epoch])
			save_statistics(experimentname,[means])
			save_statistics(experimentname,[stds])
			save_statistics(experimentname,[ci95])
			




