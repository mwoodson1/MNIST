import numpy as np
import data
import chainer
import chainer.functions as F
from chainer import optimizers

mnist = data.load_mnist_data()

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])

N_test = y_test.size

x_train = x_train.reshape((N,28,28))
x_test = x_test.reshape((N_test,28,28))

n_units = 100

#Create the model
model = chainer.FunctionSet(l1=F.Convolution2D(1,32,5),
							l2=F.Convolution2D(32,32,3),
							l3=F.Convolution2D(32,16,3),
							l4=F.Linear(1296,128),
							l5=F.Linear(128,10))
'''
model = chainer.FunctionSet(l1=F.Linear(784, n_units),
							l2=F.Linear(n_units, n_units),
							l3=F.Linear(n_units, 10))
'''
def forward(x_data, y_data, train=True):
	tmp =  x_data.shape
	x_data = x_data.reshape(tmp[0],1,tmp[1],tmp[2])
	x, t = chainer.Variable(x_data), chainer.Variable(y_data)

	'''
	print "below is the type of x"
	print type(x_data)
	'''
	
	h1 = F.relu(model.l1(x))

	h2 = F.relu(model.l2(h1))
	h2 = F.dropout(F.max_pooling_2d(h2,2), train=train)

	h3 = F.dropout(F.relu(model.l3(h2)))
	#h3 = F.reshape(h3,(16*196))

	h4 = F.dropout(F.relu(model.l4(h3)), train=train)

	y = model.l5(h4)
	return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

'''
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
'''

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

batchsize = 100
for epoch in xrange(10):
	print 'epoch', epoch
	indexes = np.random.permutation(60000)
	for i in xrange(0, 60000, batchsize):
		x_batch = x_train[indexes[i : i + batchsize]]
		y_batch = y_train[indexes[i : i + batchsize]]

		optimizer.zero_grads()
		loss, accuracy = forward(x_batch, y_batch)
		loss.backward()
		optimizer.update()

sum_loss, sum_accuracy = 0, 0
for i in xrange(0, 10000, batchsize):
	x_batch = x_test[i : i + batchsize]
	y_batch = y_test[i : i + batchsize]
	loss, accuracy = forward(x_batch, y_batch)
	sum_loss      += loss.data * batchsize
	sum_accuracy  += accuracy.data * batchsize

mean_loss     = sum_loss / 10000
mean_accuracy = sum_accuracy / 10000

print mean_accuracy
print mean_accuracy

print "everything good so far"