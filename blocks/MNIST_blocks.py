from theano import tensor
x = tensor.matrix('features')

from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalActivation
input_to_hidden = Convolutional((5,5), 32, 1,border_mode='same')
h = Rectifier().apply(input_to_hidden.apply(x))


hidden_to_output = Linear(name='hidden_to_output', input_dim=100, output_dim=10)
y_hat = Softmax().apply(hidden_to_output.apply(h))

y = tensor.lmatrix('targets')
from blocks.bricks.cost import CategoricalCrossEntropy
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

from blocks.bricks import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
cg = ComputationGraph(cost)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

from blocks.bricks import MLP
mlp = MLP(activations=[Rectifier(), Softmax()], dims=[784, 100, 10]).apply(x)

from blocks.initialization import IsotropicGaussian, Constant
input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

from fuel.datasets import MNIST
mnist = MNIST("train",)

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
data_stream = Flatten(DataStream.default_stream(
 mnist,
 iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))

from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(step_rule=None,cost=cost,params=cg.parameters)

mnist_test = MNIST("test",)
data_stream_test = Flatten(DataStream.default_stream(
 mnist_test,
 iteration_scheme=SequentialScheme(
     mnist_test.num_examples, batch_size=1024)))

from blocks.extensions.monitoring import DataStreamMonitoring
monitor = DataStreamMonitoring(
 variables=[cost], data_stream=data_stream_test, prefix="test")

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                  extensions=[monitor, FinishAfter(after_n_epochs=1), Printing()])
main_loop.run() # doctest: +SKIP
