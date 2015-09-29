import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report

mnist = fetch_mldata('mnist-original')
X_train, X_test, y_train, y_test = train_test_split(
		(mnist.data / 255.0).astype(np.float32),
		mnist.target.astype(np.int32),
		test_size=1.0/7.0, random_state=1234)

from sknn.mlp import Classifier, Convolution, Layer

nn = Classifier(
	layers=[
		Convolution("Rectifier", channels=32, kernel_shape=(5,5)),
		Convolution("Rectifier", channels=32, kernel_shape=(3,3), pool_shape=(2,2), dropout=0.5),
		Convolution("Rectifier", channels=16, kernel_shape=(3,3), dropout=0.5),
		Layer("Rectifier", units=128, dropout=0.5),
		Layer("Softmax")],
	learning_rate=0.02,
	learning_rule='adadelta,
	n_iter=10)

print "Fitting the model"
nn.fit(X_train, y_train)

print "Model fitted"
y_pred = nn.pred(X_test)

print "Predictions are:"
print y_pred

print "Actual values are:"
print y_test