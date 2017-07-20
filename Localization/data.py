from random import sample
class Data:

    def __init__(self, x,label):
        self.x=x
        self.label = label


    def batch(self, batch_size):
        randList= sample(xrange(len(self.x)),batch_size)
        x=[]
        y=[]
        for i in range(0,batch_size):
            x.append(self.x[randList[i]])
            y.append(self.label[randList[i]])
        return Data(x,y)
