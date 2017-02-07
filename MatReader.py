from scipy import io
for i in range(4):
    dataDict = io.loadmat('res/dataset1.mat')
    print('dataDict', dataDict)

