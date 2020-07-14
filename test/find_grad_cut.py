import sys
import numpy as np

top_dict = {}

def list_updata(pic_id, id_2, dis, length):
  if(pic_id not in top_dict):
    top_dict[pic_id] = [dis]
  else:
    top_dict[pic_id].append(dis)



def build_top(csv_file, length):
  with open(csv_file) as f:
    for line in f:
      if('SCORE' in line):
        continue
      line = line.strip()
      eles = line.split(',')
      feature_dis = float(eles[2])
      id_1 = int(eles[0])
      id_2 = int(eles[1])
      list_updata(id_1, id_2, feature_dis, length)
      list_updata(id_2, id_1, feature_dis, length)

def find_cut_off_grad(top_dict):
  for item in top_dict:
    item_list = top_dict[item]
    item_list.sort()
    item_list.reverse()
    top_dict[item] = (0.0, 0.0)
    for i in range(len(item_list) - 1):
      grad = item_list[i] - item_list[i + 1]
      if(grad > 0.1):
        top_dict[item] = (item_list[i+1], grad)
      continue
        
  
csv_file = sys.argv[1]
build_top(csv_file, 20)
find_cut_off_grad(top_dict)
      
with open('grad_cut.csv', 'w') as w:
  with open(csv_file) as f:
    for line in f:
      if('SCORE' in line):
        w.write(line)
        continue
      line = line.strip()
      eles = line.split(',')
      feature_dis = float(eles[2])
      id_1 = int(eles[0])
      id_2 = int(eles[1])
      thres_1, grad_1 = top_dict[id_1]
      thres_2, grad_2 = top_dict[id_2]
      final_dis = feature_dis
      if(feature_dis > thres_1 + 1e-3):
        final_dis += grad_1 * 0.1
      else:
        final_dis -= grad_1 * 0.1
      if(feature_dis > thres_2 + 1e-3):
        final_dis += grad_2 * 0.1
      else:
        final_dis -= grad_2 * 0.1

      #final_dis = feature_dis
      
      
      
      new_line = str(id_1) + "," + str(id_2) + "," + str(round(final_dis, 5)) + '\n'
      w.write(new_line)
