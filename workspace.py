import numpy
from neuralNetwork import neuralNetwork
import matplotlib.pyplot
import matplotlib.image
from random import randint

def starttraining():
    # load MNIST training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
        # create the target output valies (all 0.1, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)
        pass
    pass

def testnetwork(index):
    # load the MNIST test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # get the first test record
    all_values = test_data_list[index].split(',')
    # get the guess from the network
    netanswer = net.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    # print the label
    print "Network answer:"
    print netanswer
    print "Correct answer:"
    print numpy.argmax(all_values[0])+1
    # show the image
    showimage(all_values)
    pass

def showimage(all_values):
    # reshape the received data to an image array
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    # show this data in a new window
    matplotlib.pyplot.close()
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    pass

matplotlib.pyplot.ion()
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# learning rate is 0.3
learning_rate = 0.3
# return instance of neural network
net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train the net
starttraining()

# test the net
testnetwork(randint(0, 9))
