from math import *
from matplotlib import pyplot 
from numpy import *
def loadData1():
    data = [];label = []
    file = open("/home/xunw/data/logistData.txt")
    lines = file.readlines()
    for line in lines:
        lineArr = line.strip().split()
        data.append([1.0,float(lineArr[0]),float(lineArr[1])])
        label.append(int(lineArr[2]))
    return data,label
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     # m*n   matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadData1()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    

    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
def plot2():
    data,label = loadData1()
    weights= gradAscent(data,label)
    print weights
#     weights = ones((3,1))
#     weights[0]=4.12414349
#     weights[1] = 0.48007329
#     weights[2] =  -0.6168482
#     print weights
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    pyplot.plot(x,y)
    pyplot.show()
# def loadMnist():
#     file = open("/home/xunw/data/t10k-images.idx3-ubyte")
#     for i in range(100):
#         line = file.readline()
#         print line
#     return    


wang = 1
def printx():
    print wang

if __name__ == "__main__":
  
    print 'main'
#     data,label = loadData1()
#     w = gradAscent(data,label)
    

