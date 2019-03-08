#encoding=utf-8
#Author: CJ
#Date: 2019-03-06 11:20:00

class Record:

    def __init__(self):
        
        self.clear()

    def update(self, tmp_chunk_correct, tmp_total_true, tmp_total_pred, tmp_loss):
        self.chunk_correct += tmp_chunk_correct
        self.total_true += tmp_total_true
        self.total_pred += tmp_total_pred
        self.loss += tmp_loss
    
    def show_overall(self, epoch):
        precision = self.chunk_correct / self.total_pred if self.chunk_correct > 0 else 0
        recall = self.chunk_correct / self.total_true if self.chunk_correct > 0 else 0
        f1 = 2 * precision * recall / (recall + precision) if precision > 0 else 0
        print "%d th epoch chunk precision: %f" % (epoch, precision)
        print "%d th epoch chunk recall: %f" % (epoch, recall)
        print "%d th epoch chunk f1: %f" % (epoch, f1)
        print "%d th epoch Average loss: %f" % (epoch, self.loss)

    def show_category(self, epoch, count_dic):
        print '-------- %dth epoch --------' % epoch
        for key, count_list in count_dic.items():
            recall = count_list[0] * 1.0 / count_list[2]
            precision = count_list[0] * 1.0 / count_list[1]
            f1 = 2 * precision * recall / (recall + precision)
            print '%s precision: %.2f; recall: %.2f; f1: %.2f' % (key, precision, recall, f1)
        return

    def clear(self):
        self.chunk_correct = 0.0
        self.total_true = 0.0
        self.total_pred = 0.0
        self.loss = 0.0
