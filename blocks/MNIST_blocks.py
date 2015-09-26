from theano import tensor
x = tensor.matrix('features')

from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.bricks import Linear, Rectifier, Softmax, Tanh

from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter

from blocks.bricks import MLP

from blocks.initialization import IsotropicGaussian, Constant


from fuel.datasets import MNIST


from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


from blocks.algorithms import GradientDescent, Scale

mlp = MLP(activations=[Tanh(), Softmax()], dims=[784, 100, 10],
		  weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
mlp.initialize()

x = tensor.matrix('features')
y = tensor.matrix('targets')
y_hat = mlp.apply(x)
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
error_state = MissclassificationRate().apply(y.flatten(), y_hat)

mnist_train = MNIST(("train",))
train_stream = Flatten(
	DataStream.default_stream(
		dataset=mnist_train,
		iteration_scheme=SequentialScheme(mnist_train.num_examples, 128)),
	which_sources=('features',))