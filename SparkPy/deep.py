from math import *
from numpy import linalg as LA
from matplotlib import pyplot 
from numpy import *
from os import listdir
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def sigdr(inx):   #input Mat 
    return multiply(sigmoid(inx),(1-sigmoid(inx)))
def img2Vector(filename):
    vec = zeros((1,32*32))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vec[0,32*i + j] = int(line[j])
    return vec   
# vec = img2Vector("/home/xunw/data/digits/trainingDigits/0_0.txt")
# print vec[0,:100]
def loadFile(file):  #img 32*32  0-9digit
    dir = file
    files = listdir(dir)
    m =  len(files)
    trainMat = zeros((m,1024))
    trainLabel = zeros((m,10))
    for i in range(10):
        fileName = files[i]
        print fileName
        trainLabel[i,int(files[i].split('_')[0])] = 1
        print trainLabel[i].argmax()
        trainMat[i,:] = img2Vector(dir+'/'+files[i])
    print shape(trainMat),shape(trainLabel)
    return trainMat,trainLabel

def loadAndSequential():
    trainFile = "/home/xunw/data/digits/trainingDigits"
    testFile = "/home/xunw/data/digits/testDigits"
    trainData,trainLabel = loadFile(trainFile)
    testData,testLabel = loadFile(testFile)
#     save('trainMat.npy',trainData)
#     save('trainLabel.npy',trainLabel)
#     save('testMat.npy',testData)
#     save('testLabel.npy',testLabel)
#loadAndSequential()   readData file file

#----------------------------------------------------------------------------------------------------------------
trainMat = load('trainMat.npy')  #(1934, 1024)
trainLabel = load('trainLabel.npy')   #(1934, 10)
testMat = load('testMat.npy')    #(946, 1024)
testLabel = load('testLabel.npy')# (946, 10)


mean,stddeviation= 0,0.1
x = mat(trainMat).T  #1024*1934
y = mat(trainLabel).T     #10*1934
n1 = 1024 ; n2 = 300;n3 = 10;
nn1 = random.normal(0,0.1,n2*n1)
W1 = mat(nn1.reshape(n2,n1))
b1 = mat(random.normal(0,0.1,n2)).T
print b1
nn2 = random.normal(0,0.1,n2*n3)
W2= mat(nn2.reshape(n3,n2))   #(10,100)
b2 = mat(random.normal(0,0.1,n3)).T
m = 1934
alpha = 0.03
lamd = 0.0

def fp():
    z2 = W1*x + b1   #100 *1 => 100*1934
    a2 = sigmoid(z2) 
    z3 = W2*a2+b2    #10 *1 =>10*1934
    h = sigmoid(z3)  #10*1934
    
    def bp():
        grandW1 = zeros(shape(W1))
        grandb1 = zeros((n2,1)) ;
        grandW2 = zeros(shape(W2))
        grandb2 = zeros((n3,1)) ;
        for i in range(m):
            delta3 = multiply(-(y[:,i]-h[:,i]),sigdr(z3[:,i]))
            #print shape(W2.T),shape(delta3[:,i]),shape(sigdr(z2[:,i]))
            delta2 = multiply(W2.T*delta3,sigdr(z2[:,i]))
            grandW2 = grandW2+delta3*a2[:,i].T
            grandb2 = grandb2+delta3
            grandW1 = grandW1+delta2*x[:,i].T
            grandb1 = grandb1+delta2
        return grandW1,grandW2,grandb1,grandb2
    grandW1,grandW2,grandb1,grandb2 = bp()
    global W1,W2,b1,b2 
    W1 =  W1 - alpha*(grandW1/m+lamd*W1)
    W2 =  W2 - alpha*(grandW2/m+lamd*W2)
    b1 =  b1 - alpha*grandb1/m
    b2 =  b2 - alpha*grandb2/m  
    errCost = cost(h,trainLabel)
    return errCost

def cost(h,trainLabel):  #h  10*1934
    distMat = h.T - trainLabel#1934*10
    err = 0
    for i in range(1934):
        err += LA.norm(distMat[i].A[0])
    return err
def deeplearning():
     
    errOld = fp()
    errNew = fp()
    while abs(errNew - errOld) > 0.2:
        print 'err dist ',errNew - errOld
        errOld = errNew
        #print b2.T
        errNew = fp()
        print 'err ' ,errNew
        
    save('W1.npy',W1)
    save('W2.npy',W2)
    save('b1.npy',b1)
    save('b2.npy',b2)
        
def test():
    ww1 = load('W1.npy')
    ww2 = load('W2.npy')
    bb1 = load('b1.npy')
    bb2 = load('b2.npy')
    tX =mat(testMat).T    #testMat
    testNum = 946
    z2 = ww1*tX + bb1   #100 *1 => 100*946
    a2 = sigmoid(z2) 
    z3 = ww2*a2+bb2    #10 *1 =>10*946
    h = sigmoid(z3)  #10*946
    Y = h.T #946*10
    error = 0;
    for i in range(testNum):
        trueLabel = testLabel[i].argmax()    #testLabel
        y=Y[i].A[0]
        print trueLabel,y.argmax()
        predictLabel = y.argmax()
        error += (trueLabel!=predictLabel)
    print float(error)/testNum    
        
if __name__ == "__main__":
    print 'start...'
    deeplearning()
    test()
    


