import sys

with open(sys.argv[1]) as f:
  with open(sys.argv[2]) as t:
    line_s_1 = f.readlines()
    line_s_2 = t.readlines()
    with open('predictions.csv','w') as w:
      for i in range(len(line_s_1)):
        if(i == 0):
          w.write(line_s_1[i])
          continue
        line1 = line_s_1[i].strip()
        line2 = line_s_2[i].strip()
        eles = line1.split(',')
        id1 = eles[0]
        id2 = eles[1]     
        score_1 = float(eles[2])
        score_2 = float(line2.split(',')[2])
        if(score_1 >0.99):
          score_3 = score_1
        else:
          score_3 = score_2
        w_line = id1 + ',' + id2 + ',' + str(score_3) + '\n'
        w.write(w_line)
      
