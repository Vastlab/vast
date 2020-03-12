import numpy as np
import os
import cv2
from sklearn.utils import shuffle


### (1) preparing lable to int & int to label dictionary
full_dic_int_to_label = {}
with open("imagenet1000_clsidx_to_labels.txt") as f:
  for line in f:
    (val, key) = line.split(': ')
    key = line.split("'")[1]
    #print("val = ", val, "\n")
    #print("key = ", key, "\n\n\n")
    full_dic_int_to_label[int(val)] = key

known_dic_int_to_label = {k: full_dic_int_to_label[k] for k in range(300)}
known_unknown_dic_int_to_label = {k: full_dic_int_to_label[k] for k in range(300,600)}
unknown_dic_int_to_label = {k: full_dic_int_to_label[k] for k in range(600,1000)}

full_dic_label_to_int = {v: k for k, v in full_dic_int_to_label.items()}
known_dic_label_to_int = {v: k for k, v in known_dic_int_to_label.items()}
known_unknown_dic_label_to_int = {v: k for k, v in known_unknown_dic_int_to_label.items()}
unknown_dic_label_to_int = {v: k for k, v in unknown_dic_int_to_label.items()}



### (2) dictionary from train folder number to class number (1 to 1000)
temp = 0;
imagenet_train_all_folders_dic = {}
with open("tarin_label.txt") as f:
  for line in f:
    temp = temp + 1
    val = line[1:9]
    key = line[10:-1]
    #print("val = ", val, "\n")
    #print("key = ", key, "\n\n\n")
    #imagenet_train_all_files_dic[int(val)] = key
    imagenet_train_all_folders_dic[int(val)] = full_dic_label_to_int[key]

#print(imagenet_train_all_files_dic)
known_folders_train_dic = {k: v for k, v in imagenet_train_all_folders_dic.items() if v<300}
known_unknow_folders_train_dic = {k: v for k, v in imagenet_train_all_folders_dic.items() if (299<v and v<600)}
unknown_folders_train_dic = {k: v for k, v in imagenet_train_all_folders_dic.items() if v>599}



'''
imagenet_val_all_files_dic = {}
with open("val_lable.txt") as f:
  val = 0
  for line in f:
    val = val + 1
    key = int(line[:-1])
    #print("val = ", val, "\n")
    #print("key = ", key, "\n\n\n")
    imagenet_val_all_files_dic[int(val)] = key
'''



path_train = '/net/kato/datasets/ImageNet/ILSVRC_2012/train'


### (3) preparing known data 

known_keys = known_folders_train_dic.keys()
known_train_full_path = []
known_train_label = []
known_test_full_path = []
known_test_label = []
known_test_val_full_path =  []
known_test_val_label =  []
known_test_test_full_path = []
known_test_test_label =  []

for foldernumber in  known_keys:
  v = known_folders_train_dic[foldernumber] # v = class number for folder_number 
  tmp_path = path_train + '/n' + str(foldernumber).zfill(8)
  filenames= os.listdir (tmp_path)
  nnn = 0
  for filename in filenames:
    full_path = tmp_path + '/' + filename
    #print(full_path)
    if nnn <300 :
      known_test_full_path.append( full_path )
      known_test_label.append( v )
      if nnn < 50 :
        known_test_val_full_path.append( full_path )
        known_test_val_label.append( v )
      else:
        known_test_test_full_path.append( full_path )
        known_test_test_label.append( v )
    else:
      known_train_full_path.append( full_path )
      known_train_label.append( v )
    nnn = nnn + 1
  if nnn < 400:
    print("Error: number of known data is " , nnn , " for class v", v)

    
known_train_full_path , known_train_label = shuffle(known_train_full_path , known_train_label)
known_test_full_path , known_test_label = shuffle(known_test_full_path , known_test_label)
known_test_val_full_path , known_test_val_label = shuffle(known_test_val_full_path , known_test_val_label) 
known_test_test_full_path , known_test_test_label = shuffle(known_test_test_full_path , known_test_test_label) 


### (4) preparing known unknown data

known_unkonwn_keys = known_unknow_folders_train_dic.keys()
known_unknown_train_full_path = []
known_unknown_train_label = []
known_unknown_test_full_path = []
known_unknown_test_label = []
known_unknown_test_val_full_path =  []
known_unknown_test_val_label =  []
known_unknown_test_test_full_path = []
known_unknown_test_test_label =  []

for foldernumber in  known_unkonwn_keys:
  v = known_unknow_folders_train_dic[foldernumber]
  tmp_path = path_train + '/n' + str(foldernumber).zfill(8)
  filenames= os.listdir (tmp_path)
  nnn = 0
  for filename in filenames:
    full_path = tmp_path + '/' + filename
    if nnn< 300:
      known_unknown_test_full_path.append( full_path )
      known_unknown_test_label.append( 300 )
      if nnn <50:
        known_unknown_test_val_full_path.append( full_path )
        known_unknown_test_val_label.append( 300 )
      else:
        known_unknown_test_test_full_path.append( full_path )
        known_unknown_test_test_label.append( 300 )
    else:
      known_unknown_train_full_path.append( full_path )
      known_unknown_train_label.append( 300 )
    nnn = nnn + 1
  if nnn < 400:
    print("Error: number of known-unknown data is " , nnn , " for class v", v)


known_unknown_train_full_path, known_unknown_train_label = shuffle(known_unknown_train_full_path, known_unknown_train_label)
known_unknown_test_full_path, known_unknown_test_label = shuffle(known_unknown_test_full_path, known_unknown_test_label)
known_unknown_test_val_full_path , known_unknown_test_val_label = shuffle(known_unknown_test_val_full_path , known_unknown_test_val_label) 
known_unknown_test_test_full_path , known_unknown_test_test_label = shuffle(known_unknown_test_test_full_path , known_unknown_test_test_label)



### (5) preparing unknown data

unknown_keys = unknown_folders_train_dic.keys()
unknown_train_full_path = []
unknown_train_label = []
unknown_test_full_path = []
unknown_test_label = []
unknown_test_val_full_path =  []
unknown_test_val_label =  []
unknown_test_test_full_path = []
unknown_test_test_label =  []

for foldernumber in  unknown_keys:
  v = unknown_folders_train_dic[foldernumber]
  tmp_path = path_train + '/n' + str(foldernumber).zfill(8)
  filenames= os.listdir (tmp_path)
  nnn = 0
  for filename in filenames:
    full_path = tmp_path + '/' + filename
    unknown_test_full_path.append( full_path )
    unknown_test_label.append(300)
    if nnn <50:
      unknown_test_val_full_path.append( full_path )
      unknown_test_val_label.append(300)
    else:
      unknown_test_test_full_path.append( full_path )
      unknown_test_test_label.append(300)
    nnn = nnn + 1

unknown_test_full_path , unknown_test_label = shuffle(unknown_test_full_path , unknown_test_label)
unknown_test_val_full_path , unknown_test_val_label = shuffle(unknown_test_val_full_path , unknown_test_val_label)
unknown_test_test_full_path , unknown_test_test_label = shuffle(unknown_test_test_full_path , unknown_test_test_label)

'''
path_val = '/net/kato/datasets/ImageNet/ILSVRC_2012/val/'
known_val_full_path =[]
known_val_label = []
known_unknown_val_full_path =[]
known_unknown_val_label = []
unknown_val_full_path =[]
unknown_val_label = []
for x in range(1,50001):
  v = imagenet_val_all_files_dic[x]
  full_path = path_val + 'ILSVRC2012_val_' + str(x).zfill(8) + '.JPEG'
  if v<300:
    known_val_full_path.append(full_path)
    known_val_label.append(v)
  elif v>599:
    unknown_val_full_path.append(full_path)
    unknown_val_label.append(v)
  else:
    known_unknown_val_full_path.append(full_path)
    known_unknown_val_label.append(v)
'''


with open('known_train_full_path.txt', 'w') as file_handler:
  for item in known_train_full_path:
    file_handler.write("{}\n".format(item))

with open('known_train_label.txt', 'w') as file_handler:
  for item in known_train_label:
    file_handler.write("{}\n".format(item))

with open('known_unknown_train_full_path.txt', 'w') as file_handler:
  for item in known_unknown_train_full_path:
    file_handler.write("{}\n".format(item))

with open('known_unknown_train_label.txt', 'w') as file_handler:
  for item in known_unknown_train_label:
    file_handler.write("{}\n".format(item))


with open('known_test_full_path.txt', 'w') as file_handler:
  for item in known_test_full_path:
    file_handler.write("{}\n".format(item))

with open('known_test_label.txt', 'w') as file_handler:
  for item in known_test_label:
    file_handler.write("{}\n".format(item))

with open('known_unknown_test_full_path.txt', 'w') as file_handler:
  for item in known_unknown_test_full_path:
    file_handler.write("{}\n".format(item))

with open('known_unknown_test_label.txt', 'w') as file_handler:
  for item in known_unknown_test_label:
    file_handler.write("{}\n".format(item))

with open('unknown_test_full_path.txt', 'w') as file_handler:
  for item in unknown_test_full_path:
    file_handler.write("{}\n".format(item))

with open('unknown_test_label.txt', 'w') as file_handler:
  for item in unknown_test_label:
    file_handler.write("{}\n".format(item))


############## 
    
    
with open('known_test_val_full_path.txt', 'w') as file_handler:
  for item in known_test_val_full_path:
    file_handler.write("{}\n".format(item))

with open('known_test_val_label.txt', 'w') as file_handler:
  for item in known_test_val_label:
    file_handler.write("{}\n".format(item))

with open('known_test_test_full_path.txt', 'w') as file_handler:
  for item in known_test_test_full_path:
    file_handler.write("{}\n".format(item))

with open('known_test_test_label.txt', 'w') as file_handler:
  for item in known_test_test_label:
    file_handler.write("{}\n".format(item))    

    
    
with open('known_unknown_test_val_full_path.txt', 'w') as file_handler:
  for item in known_unknown_test_val_full_path:
    file_handler.write("{}\n".format(item))

with open('known_unknown_test_val_label.txt', 'w') as file_handler:
  for item in known_unknown_test_val_label:
    file_handler.write("{}\n".format(item))

with open('known_unknown_test_test_full_path.txt', 'w') as file_handler:
  for item in known_unknown_test_test_full_path:
    file_handler.write("{}\n".format(item))

with open('known_unknown_test_test_label.txt', 'w') as file_handler:
  for item in known_unknown_test_test_label:
    file_handler.write("{}\n".format(item))        
    

    
    
with open('unknown_test_val_full_path.txt', 'w') as file_handler:
  for item in unknown_test_val_full_path:
    file_handler.write("{}\n".format(item))

with open('unknown_test_val_label.txt', 'w') as file_handler:
  for item in unknown_test_val_label:
    file_handler.write("{}\n".format(item))

with open('unknown_test_test_full_path.txt', 'w') as file_handler:
  for item in unknown_test_test_full_path:
    file_handler.write("{}\n".format(item))

with open('unknown_test_test_label.txt', 'w') as file_handler:
  for item in unknown_test_test_label:
    file_handler.write("{}\n".format(item))        
    
    
'''
with open('known_val_full_path.txt', 'w') as file_handler:
  for item in known_val_full_path:
    file_handler.write("{}\n".format(item))
with open('known_val_label.txt', 'w') as file_handler:
  for item in known_val_label:
    file_handler.write("{}\n".format(item))
with open('known_unknown_val_full_path.txt', 'w') as file_handler:
  for item in known_unknown_val_full_path:
    file_handler.write("{}\n".format(item))
with open('known_unknown_val_label.txt', 'w') as file_handler:
  for item in known_unknown_val_label:
    file_handler.write("{}\n".format(item))
with open('unknown_val_full_path.txt', 'w') as file_handler:
  for item in unknown_val_full_path:
    file_handler.write("{}\n".format(item))
with open('unknown_val_label.txt', 'w') as file_handler:
  for item in unknown_val_label:
    file_handler.write("{}\n".format(item))
'''
