import numpy as np
import tqdm
from storage import *
import os
import time
from A3_B1_C4_1shot import *
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

experimentname = 'A3_B1_C4_1shot_mit'

for test_epoch in range(1):
	model = GCNMetaModel(myconfig)
	log_to_save = []
	total_time_start = time.time()
	save_statistics(experimentname,['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'best_val_acc'])
	(best_val_acc, best_val_acc_epoch) = (0.0, 0)
	for epoch in range(300):
		print("=={} {}".format(experimentname,epoch))
		train_loss, train_acc, train_speed, train95 = model.run_epoch(is_training = True)
		print("\r\x1b[K Train: loss: %.5f | acc: %.5f | instances/sec: %.2f | ui95: %.5f" % (train_loss, train_acc, train_speed, train95))
		valid_loss, valid_acc, valid_speed, val95 = model.run_epoch(is_training = False)
		print("\r\x1b[K Valid: loss: %.5f | acc: %.5f | instances/sec: %.2f | ui95: %.5f" % (valid_loss, valid_acc, valid_speed, val95))

		epoch_time = time.time() - total_time_start
		
		val_acc = valid_acc  # type: float
		if val_acc > best_val_acc:
		#	self.save_model(best_model_file)
			print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. )" % (val_acc, best_val_acc))
			best_val_acc = val_acc
			
		print("\r\x1b[K best acc: %.5f" % (best_val_acc))
		
		save_statistics(experimentname,[epoch, train_loss, train_acc, valid_loss, valid_acc, best_val_acc])




