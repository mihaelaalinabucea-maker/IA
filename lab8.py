import numpy as np
from numpy.ma.core import identity

a=np.array([1,2,3])
print(type(a))
print(a.dtype)
print(a.shape)
print(a[0])


b=np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0][2])
print(b[0,2])

c=np.asarray([[1,2],[3,4]])
print(type(c))
print(c.shape)



zero_array=np.zeros((3,2))
print (zero_array)
ones_array=np.ones((2,2))
print (ones_array)
constant_array=np.full((2,2),8)
print (constant_array)
identity_matrix=np.eye(3)
print (identity_matrix)
random_array=np.random.randn(1,2)
print(random_array)
mu, sigma=0,0.1
gaussian_random=np.random.normal(mu,sigma,(3,6))
first_5=np.arange(5)
print(first_5)

first_5=np.arange(5)
print(first_5)
array_to_slice=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
slice=array_to_slice[:,0:3]
print(slice)
print(array_to_slice[0][0])
slice[0][0]=100
print(array_to_slice[0][0])
slice_copy=np.copy(array_to_slice[:,0:3])
slice_copy[0][0]=100
print(slice_copy[0][0])
print(array_to_slice[0][0])


x=np.array([[1,2],[3,4]],dtype=np.float64)
y=np.array([[5,6],[7,8]],dtype=np.float64)
print(x+y)
print(np.add(x,y))
print(x-y)
print(np.subtract(x,y))
print(x*y)
print(np.multiply(x,y))
print(x/y)
print(np.divide(x,y))
print(np.sqrt(x))
my_array=np.arange(5)
powered=np.power(my_array,3)
print(powered)

x=np.array([[1,2],[4,5]])
y=np.array([[5,6],[7,8]])
v=np.array([9,10])
w=np.array([11,12])

print(v.dot(w))
print(np.dot(v,w))
print(np.matmul(x,v))
print(np.matmul(x,y))

my_array=np.array([[1,2,3],[4,5,6]])
print(my_array.T)
my_array=np.array([[1.,2.],[3.,4.]])
print(np.linalg.inv(my_array))

x=np.array([[1,2],[3,4]])
print(np.sum(x))
print(np.sum(x,axis=0))
print(np.sum(x,axis=1))
print(np.sum(x,axis=(0,1)))


y=np.array([[[1,2,3,4],[5,6,7,8]],[[1,2,3,4],[5,6,7,8]],[[1,2,3,4],[5,6,7,8]]])
print(y.shape)
print(y)
print(np.mean(y,axis=0))
print(np.mean(y,axis=1))
z=np.array([[10,12,5],[17,11,19]])
print(np.argmax(z,axis=1))


