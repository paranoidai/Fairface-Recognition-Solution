import numpy as np
import mxnet as mx
import pdb

class AccMetric(mx.metric.EvalMetric):
  def __init__(self, acc_name = 'acc', label_index = 0, pred_index = 1):
    self.axis = 1
    super(AccMetric, self).__init__(
        acc_name, axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0
    self.label_index = label_index
    self.pred_index = pred_index

  def update(self, labels, preds):
    self.count+=1
    #print(labels)
    label = labels[0]
    pred_label = preds[self.pred_index]
    #print('pre_label.shape', pred_label.shape)
    #print(pred_label)
    #item = preds[-2].asnumpy()
    #for item in pred_label.asnumpy():
    #    print(item)
    #print('pred_logits', np.max(item), np.min(item), np.min(np.abs(item)))
    #print('pred_label:', pred_label.shape)
    #exit()
    #attr_pred_label = preds[2]
    #attr_softmax_loss = preds[4]
    #print("soft_softmax, ", preds[5][0:3])
    #print("hard_softmax, ", preds[4][0:3])
    #print('embeddings: ', preds[0] , 'attr_pred_label: ', attr_pred_label, 'attr_softmax_loss: ', attr_softmax_loss)
    #pdb.set_trace()
    #print(pred_label.asnumpy())
    #return
    #logits = preds[2].asnumpy()
    #print(np.max(logits))
    #print('ACC', label.shape, pred_label.shape)
    #if pred_label.shape != label.shape:
    pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
    #print('axis, ', self.axis)
    #print('argmax shape', pred_label.shape)
    pred_label = pred_label.asnumpy().astype('int32').flatten()
    label = label.asnumpy()
    if label.ndim==2:
      label = label[:,self.label_index]
    #print('slice_label', label)
    label = label.astype('int32').flatten()
    #print('lllllllabel, ', label.shape, pred_label.shape)
    assert label.shape==pred_label.shape
    for i in len(label):
      if(label[i] != pred_label[i]):
        print(label_index)
    self.sum_metric += (pred_label.flat == label.flat).sum()
    self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    #label = labels[0].asnumpy()
    pred = preds[-1].asnumpy()
    #sn = preds[-2].asnumpy()
    #print('sssssssssn', sn.shape)
    #sp = preds[-3].asnumpy()
    #print('ssssssssp', sp.shape)
    #print('in loss', pred.shape)
    #print(pred)
    loss = pred[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    #gt_label = preds[-2].asnumpy()
    #print(gt_label)
