from pybrain.structure import FeedForwardNetwork, SigmoidLayer, LinearLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np

net = FeedForwardNetwork()

l_in = LinearLayer(4)
l_hid = SigmoidLayer(6)
l_out = LinearLayer(3)

net.addInputModule(l_in)
net.addModule(l_hid)
net.addOutputModule(l_out)

i2h = FullConnection(l_in, l_hid)
h2o = FullConnection(l_hid, l_out)

net.addConnection(i2h)
net.addConnection(h2o)

net.sortModules()
print net

dataset = SupervisedDataSet(4,3)
dataset.addSample((1, 0, 0, 0), (1, 0, 0))
dataset.addSample((0, 1, 0, 0), (0, 1, 0))
dataset.addSample((0, 0, 0, 1), (0, 0, 1))
dataset.addSample((0, 0, 1, 1), (0, 0, 1))
dataset.addSample((0, 0, 1, 0), (0, 1, 0))
dataset.addSample((1, 1, 0, 0), (1, 0, 0))

trainer = BackpropTrainer(net, dataset)
trainer.trainUntilConvergence()

a = net.activate((1,0,0,0))
b = net.activate((0,1,0,0))
c = net.activate((0,0,0,1))

print "Expected %d, predicted: %d" % (0, np.argmax(a))
print "Expected %d, predicted: %d" % (1, np.argmax(b))
print "Expected %d, predicted: %d" % (2, np.argmax(c))
