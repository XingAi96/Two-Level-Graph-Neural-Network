import os
import matplotlib.pyplot as plt


def draw(path,type):

    epoches,losses,train_accs,test_accs=[],[],[],[]
    max_acc=0
    with open(path,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.split(' ')
            epoch=float(line[0])
            loss=float(line[1])
            train_acc=float(line[2])
            test_acc=float(line[3])

            if test_acc>max_acc:
                max_acc=test_acc

            epoches.append(epoch)
            losses.append(loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        f.close()

    plt.plot(epoches, train_accs)
    plt.plot(epoches, test_accs)
    plt.plot(epoches, [max_acc]*len(epoches))
    #plt.plot(epoches, losses)
    plt.show()

check_path=r'D:\powerful-gnns-master\checkpoints\COX2\0414_1048'
for i in range(10):
    path=os.path.join(check_path,str(i)+'.txt')
    draw(path,'test')