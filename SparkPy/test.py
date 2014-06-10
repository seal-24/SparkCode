from numpy import *
import numpy
from numpy import linalg as LA
m = mat(ones((3,2)))
ma = mat(ones((3,2)))
a1 = ones(3)+1 #1,3  one-dimention
a11 = ones((1,3))
a2 = ones((1,3))+1
a3 = ones((3,1))+1  #3,1  2-dimention
def squar(inx):
    return inx*inx
print squar(a1) #[ 4.  4.  4.]
print squar(a2) #[[ 4.  4.  4.]]
print squar(a3) #[[4][4][4]]
print shape(a1),shape(a11),shape(a3)  #(3,) (1, 3) (3, 1)
print a1*a3   #not error but not expected
print a11*a3  #not matrix output
a1m = mat(ones((1,3)))
a3m = mat(ones((3,1))+1)
print a1m*a3m    #return matrix [[6]]
print shape(a1m),shape(a3)
print a1m*a3     #matrix*2-dnarray ok
print a1m[0]  # [[ 1.  1.  1.]]
print a11[0]  # [ 1.  1.  1.]
#print a1m*a1  # erroe
#print squar(a1m) error

mm = mat([[1,2,3],[1,1,3]])
print mm[0].A[0].argmax()
print LA.norm(mm[0].A[0])
print mm+mat(ones((2,1)))
