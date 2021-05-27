import time
import numpy as np
import pickle
import cv2
import PIL
import torch
import os
import json
import argparse
from collections import OrderedDict
from functools import partial
from torch import Tensor
import torch.multiprocessing as mp
from my_lib import *
from vast.opensetAlgos.EVM import EVM_Training , EVM_Inference
from vast import activations
from statistics import mean
import gc

##########################
import random
import numpy as np

def arrange_data(data, time_series_len, num_to_insert):
  data_split = []
  array_by_time_series_len = []
  count = 0
  print('NUM TO INSERT', num_to_insert)
  for run in data:
    if len(run)%time_series_len != 0:
      current = run[:-(len(run)%time_series_len)]
    else:
      current = run
    for time_step in current:
      count+=1
      array_by_time_series_len = array_by_time_series_len+time_step
      if count == time_series_len:
        count = 0
        array_by_time_series_len.insert(0, num_to_insert)
        data_split.append(array_by_time_series_len)
        array_by_time_series_len = []
  return data_split

def load_data(filenames, split_loc, end_loc, time_series_len):

  combined_train_data_split = []
  combined_test_data_split = []
  num_to_insert = 1
  for filename in filenames:
    data = np.load(filename, allow_pickle=True)
    random.shuffle(data)
    train_data = data[:split_loc]
    test_data = data[split_loc:end_loc]
    train_data_split = arrange_data(train_data, time_series_len, num_to_insert)
    test_data_split = arrange_data(test_data, time_series_len, num_to_insert)
    combined_train_data_split = combined_train_data_split + train_data_split
    combined_test_data_split = combined_test_data_split + test_data_split
    num_to_insert+=1

  random.shuffle(combined_train_data_split)
  random.shuffle(combined_test_data_split)

  return torch.from_numpy(np.asarray(combined_train_data_split)), torch.from_numpy(np.asarray(combined_test_data_split))

############################

try:
  torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
  pass


try:
  mp.set_start_method('spawn', force=True)
except RuntimeError:
  pass

number_of_classes = 2

n_cpu = int(os.cpu_count()*0.8)


def evm_train_process(classes_to_process, feature_dic, args_evm, gpu, Q, done_event):
  with torch.no_grad():
    evm_iterator_i = EVM_Training(classes_to_process, feature_dic, args_evm, gpu)
    evm_model = OrderedDict()
    n = 0
    for evm in enumerate(evm_iterator_i):
      #evm_model[evm[1][1][0]] = evm[1][1][1]
      Q.put((evm[1][1][0],evm[1][1][1]))
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
    done_event.wait()
    del evm
    return


def train_evm(feature_dic, args_evm, gpu):
  with torch.no_grad():
    t1 = time.time()
    classes_to_process = list(range(1,int(number_of_classes)+1))
    list_of_tuples = [(0,0)] * number_of_classes
    Q = mp.Queue()
    done_event = [ mp.Event()  for k in range(number_of_classes)]

    NG = min(  len(gpu) , number_of_classes  )
    assert NG > 0
    processes = []

    for k in range(number_of_classes):
      r = k % NG
      p = mp.Process(target=evm_train_process, args=([classes_to_process[k]], feature_dic, args_evm, gpu[r], Q, done_event[k]))
      p.start()
      processes.append(p)
      if r == ( NG  - 1 ):
        for n in range(NG):
          L , D = Q.get()
          list_of_tuples[L-1] = (L, D)
          done_event[L-1].set()
        for p in processes:
          p.join()
        processes.clear()
    if r !=  ( NG  - 1 ):
      for n in range(r+1):
        L , D = Q.get()
        list_of_tuples[L-1] = (L, D)
        done_event[L-1].set()
      for p in processes:
        p.join()
    evm_model = OrderedDict(list_of_tuples)
    t2 = time.time()
    print("training evm time = ", t2 - t1)
    del Q, done_event, L, D, p, list_of_tuples, processes
    return evm_model


def val_process(classes_to_process, feature_dic, evm_model, args_evm, gpu, Q, done_event):
  with torch.no_grad():
    top1_Meter = Average_Meter()
    Pr_iterator = EVM_Inference(classes_to_process, feature_dic, args_evm, gpu, evm_model)

    for k,pr in enumerate(Pr_iterator):
      r = pr[1][1].cuda(gpu)
      m, m_i  = torch.max(r, dim = 1)
      u = (1 - m).view(-1, 1)
      q = torch.cat((u, r), 1)
      norm = torch.norm(q, p=1, dim=1)
      p = q/norm[:,None]
      L = ( (pr[1][0]) * torch.ones(r.shape[0])).long().cuda(gpu)
      acc = accuracy(p, L, topk=(1, ))
      top1_Meter.update(acc[0].item(), r.size(0))
    Q.put((gpu, top1_Meter.avg, top1_Meter.count))
    done_event.wait()
    del r, m, m_i, u, q, norm, p
    del L, acc, top1_Meter, Pr_iterator



def validate(evm_model, args_evm, class_partition, feature_dic, data_name, gpu):
  with torch.no_grad():
    print("\nstart validate")
    t1 = time.time()
    NG = min(  len(gpu) , number_of_classes  )
    assert NG > 0
    classes_to_process = list(range(1,int(number_of_classes)+1))
    list_acc = [0.0] * NG
    list_count = [0] * NG
    Q = mp.Queue()
    done_event = [mp.Event() for k in range(NG)]

    process_list = []


    for k in range(NG):
      p = mp.Process(target=val_process, args=(class_partition[k], feature_dic[k], evm_model, args_evm, gpu[k], Q, done_event[k]))
      p.start()
      process_list.append(p)


    for k in range(NG):
      g, a , c = Q.get()
      print(g,a,c)
      i = gpu.index(g)
      list_acc[i] = a
      list_count[i] = c
      done_event[i].set()

    for p in process_list:
      p.join()

    print(data_name, "total accuracy = ", np.average(np.array(list_acc), weights=np.array(list_count)))

    del Q, done_event
    del p, process_list, g, a , c, list_acc, list_count
    t2 = time.time()
    print("validation time = ", t2 - t1)
    return


########################################
########################################
########################################


if __name__ == '__main__':

  print(f"Start")
  t0 = time.time()


  with open('evm_config_cosine_mini_imagenet.json', 'r') as json_file:
    evm_config = json.loads(json_file.read())
  cover_threshold = evm_config['cover_threshold']
  distance_multiplier = evm_config['distance_multiplier']
  tail_size = evm_config['tail_size']
  distance_metric =  evm_config['distance_metric']


  torch.backends.cudnn.benchmark=True


  args_evm  = argparse.Namespace()
  args_evm.cover_threshold = [cover_threshold]
  args_evm.distance_multiplier = [distance_multiplier]
  args_evm.tailsize = [tail_size]
  args_evm.distance_metric = distance_metric
  args_evm.chunk_size = 200

  ###########
  data_train, data_val = load_data(['pilco_data/100_reg.npy', 'pilco_data/100_grav_16.npy'], 98, 100, 4)
  ############

  # data_train =  torch.from_numpy(np.load("feature_train_resnet_18_mini_64.npy"))
  # data_val =  torch.from_numpy(np.load("feature_val_resnet_18_mini_64.npy"))
  filename = f"evm_{distance_metric}_f1_mini_resnet18_tail_{tail_size}_ct_{cover_threshold}_dm_{distance_multiplier}.pkl"

  # from IPython import embed; embed()

  t1 = time.time()
  print(f"loading feature time = {t1-t0}")

  t2 = time.time()

  gpu_count = torch.cuda.device_count()
  # list_of_all_gpu = list(range(gpu_count))
  list_of_all_gpu = list(range(1))
  # from IPython import embed; embed()

  features_dict_train = OrderedDict()
  for k in range(1, number_of_classes+1):
    F = data_train[data_train[:,0]==k]
    features_dict_train[k] = F[:,1:].detach().clone()



  t3 = time.time()
  print(f"computing features_dict_train time = {t3-t2}")

  # from IPython import embed; embed()

  evm_model = train_evm(features_dict_train, args_evm, list_of_all_gpu)

  t4 = time.time()
  print(f"computing train_evm time = {t4-t3}")

  print("before pickle.dump()")
  pickle.dump( evm_model, open(filename , "wb" ) )
  print("after pickle.dump()")
  del features_dict_train

  t5 = time.time()
  print(f"saving evm time = {t5-t4}")

  t5 = time.time()

  NG = len(list_of_all_gpu)
  features_dict_train = [OrderedDict() for k in range(NG)]
  class_partition = [[] for k in range(NG)]
  for k in range(1, number_of_classes+1):
    F = data_train[data_train[:,0]==k]
    r = k % NG
    class_partition[r].append(k)
    (features_dict_train[r])[k] = F[:,1:].detach().clone()


  t6 = time.time()
  print(f"computing features_dict_train time = {t6-t5}")


  evm_model = pickle.load( open( filename, "rb" ) )

  t7 = time.time()
  print(f"loading evm time = {t7-t6}")

  validate(evm_model, args_evm, class_partition, features_dict_train, 'train' , list_of_all_gpu)

  t8 = time.time()
  print(f"valiation of training data time = {t8-t7}")

  NG = len(list_of_all_gpu)
  features_dict_val = [OrderedDict() for k in range(NG)]
  class_partition = [[] for k in range(NG)]
  for k in range(1, number_of_classes+1):
    F = data_val[data_val[:,0]==k]
    r = k % NG
    class_partition[r].append(k)
    (features_dict_val[r])[k] = F[:,1:].detach().clone()

  t9 = time.time()
  print(f"computing features_dict_val time = {t9-t8}")

  validate(evm_model, args_evm, class_partition, features_dict_val, 'validation' , list_of_all_gpu)

  t10 = time.time()
  print(f"valiation of val data time = {t10-t9}")

  #print("validation accuracy = " , acc)
  t11 = time.time()
  print(f"run time = {t11-t0}")

  #from IPython import embed; embed()
