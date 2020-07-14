import sys
import numpy as np

top_dict = {}

def list_updata(pic_id, id_2, dis, length):
  if(pic_id not in top_dict):
    top_dict[pic_id] = [[pic_id],[1.0]]
  elif(len(top_dict[pic_id][0]) < length):
    top_dict[pic_id][0].append(id_2)
    top_dict[pic_id][1].append(dis)

  else:
    min_dis = min(top_dict[pic_id][1])
    if(dis > min_dis):
      replace_index = top_dict[pic_id][1].index(min_dis)
      top_dict[pic_id][1][replace_index] = dis
      top_dict[pic_id][0][replace_index] = id_2



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

def get_top_k_sim(list_a, list_b):
  top_k_sim = 0.0
  id_list_a = list_a[0]
  id_list_b = list_b[0]
  dis_list_a = list_a[1]
  dis_list_b = list_b[1]

  for item in id_list_a:
    if item in id_list_b:
      s_a = dis_list_a[id_list_a.index(item)]
      s_b = dis_list_b[id_list_b.index(item)]
      #total_s = min(s_a / s_b, s_b / s_a)
      #total_s = dis_list_a[id_list_a.index(item)] * dis_list_b[id_list_b.index(item)]
      total_s = s_a * s_b

      top_k_sim += total_s
  return top_k_sim
  
csv_file = sys.argv[1]
build_top(csv_file, 20)
      
with open('reranked.csv', 'w') as w:
  with open(sys.argv[1]) as f:
    for line in f:
      if('SCORE' in line):
        w.write(line)
        continue
      line = line.strip()
      eles = line.split(',')
      feature_dis = float(eles[2])
      id_1 = int(eles[0])
      id_2 = int(eles[1])
      
      top_k_dis = get_top_k_sim(top_dict[id_1], top_dict[id_2])
      final_dis = feature_dis * 0.5 * 1.3 + top_k_dis / 40.0 * 0.7
      new_line = str(id_1) + "," + str(id_2) + "," + str(round(final_dis, 5)) + '\n'
      w.write(new_line)
