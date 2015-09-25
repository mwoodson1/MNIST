from theano import tensor
x = tensor.matrix('features')

from blocks.bricks.cost import CategoricalCrossEntropy

print "everything ok"

##############################################3
from theano import tensor
x = tensor.matrix('features')

from blocks.bricks import Linear, Rectifier, Softmax

input_to_hidden = Linear(name="input_to_hidden", input_dim=784, output_dim=100)
h = Rectifier().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name="hidden_to_output", input_dim=100, output_dim=10)
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


from fuel.datasets import MNIST
mnist = MNIST(("train",))