import numpy
from neuralNetwork import neuralNetwork
import matplotlib.pyplot
import matplotlib.image
from random import randint
import scipy.misc

# log starting information
print("\033[1m\033[95m")
print("[NEURAL NETWORK]\033[0m")
print("\033[95m>Welcome to this basic approach of a neural network in python.\n>Type 'guide()' for help.\n>Type 'quit()' to quit.\n>Have fun! :)")
print("\033[0m")

# turn on interactive mode for pyplot
matplotlib.pyplot.ion()

# creates a net based on the parameters
def createnet(input_nodes, hidden_nodes, output_nodes, learning_rate, learning_epochs):
    global net
    net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, learning_epochs)
    # log info
    print("\033[1m\033[94m")
    print("[INITIALIZE NETWORK]\033[0m")
    print("\033[94m   stats:")
    print("  -"+str(input_nodes)+" input nodes")
    print("  -"+str(hidden_nodes)+" hidden nodes")
    print("  -"+str(output_nodes)+" output nodes")
    print("  -"+str(learning_rate)+" learning rate")
    print("  -"+str(learning_epochs)+" learning epochs.")
    print("\033[0m")
    pass

# starts the training process
def trainnet():
    if 'net' not in globals():
        __error()
        return
    # load MNIST training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # log info
    print("\033[94m\033[1m")
    print("[START TRAINING]\033[0m")
    print("\033[94m  -"+str(len(training_data_list))+" examples")
    print("  -"+str(net.epochs)+" epochs\033[0m")

    # train the neural network
    for e in range(net.epochs):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
            # create the target output valies (all 0.1, except the desired label which is 0.99)
            targets = numpy.zeros(net.onodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            net.train(inputs, targets)
            # log progress
            # print("learning number "+all_values[0]
            pass
        print("\033[92m  >Epoch #"+str(e)+" done\033[0m")
        pass
    print("\033[92m>Training complete")
    print("\033[0m")
    pass

# tests the net with all testing data
def testnet():
    if 'net' not in globals():
        __error()
        return
    # scorecard for how well the network performs, initially empty
    scorecard = []

    # load the MNIST test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # log info
    print("\033[94m\033[1m")
    print("[START TESTING]\033[0m")
    print("\033[94m  -"+str(len(test_data_list))+" test queries\033[0m")

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = net.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if(label == correct_label):
            message = "\033[92mCorrect\033[0m"
            scorecard.append(1)
        else:
            message = "\033[91mWrong\033[0m"
            scorecard.append(0)
            pass
        # print info
        # print message+" (Network: "+str(label)+" | Correct: "+str(correct_label)+")
        pass
    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    performance = (scorecard_array.sum()*1.0) / (scorecard_array.size*1.0)
    print("\033[92m>Performance: "+str(performance))
    print("\033[0m")
    pass

# tests a specific example
def test(id):
    if 'net' not in globals():
        __error()
        return
    # load the MNIST test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # log info
    print("\033[94m\033[1m")
    print("[TESTING EXAMPLE]\033[0m")
    print("\033[94m  -Testing example #"+str(id)+"\033[0m")

    # get the first test record
    all_values = test_data_list[id].split(',')
    # show the image in new window
    __showimage(numpy.asfarray(all_values[1:]))
    # get the guess from the network
    netanswer = numpy.argmax(net.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
    # print the label
    print("\033[93m  >Guess: "+str(netanswer)+"\033[0m")
    if netanswer == int(all_values[0]):
        print("\033[92m>Correct")
    else:
        print("\033[91m>Wrong")
    print("\033[0m")
    pass

# tests a random example
def testrandom():
    if 'net' not in globals():
        __error()
        return
    # load the MNIST test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # get random id
    id = randint(0, len(test_data_list)-1)

    # log info
    print("\033[94m\033[1m")
    print("[RANDOM TEST]\033[0m")
    print("\033[94m  -Testing example #"+str(id)+"\033[0m")

    # get the first test record
    all_values = test_data_list[id].split(',')
    # show the image in new window
    __showimage(numpy.asfarray(all_values[1:]))
    # get the guess from the network
    netanswer = numpy.argmax(net.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
    # print the label
    print("\033[93m  >Guess: "+str(netanswer)+"\033[0m")
    if netanswer == int(all_values[0]):
        print("\033[92m>Correct")
    else:
        print("\033[91m>Wrong")
    print("\033[0m")
    pass

# tests an image in the '/images' folder
def testimage(image_file_name):
    if 'net' not in globals():
        __error()
        return
    # get the image and reshape it as array
    img_array = scipy.misc.imread("images/"+image_file_name, flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data  =(img_data / 255.0 * 0.99) +0.01
    # show the image
    __showimage(img_data)

    # log info
    print("\033[94m\033[1m")
    print("[TEST IMAGE]\033[0m")
    print("\033[94m  -Grabbing image from 'images/"+image_file_name+"'\033[0m")

    # get the guess from the network
    netanswer = numpy.argmax(net.query(img_data))
    # print the label
    print("\033[93m>Guess: "+str(netanswer))
    print("\033[0m")
    pass

def showidea(label):
    if 'net' not in globals():
        __error()
        return

    # log info
    print("\033[94m\033[1m")
    print("[SHOWING IDEA]\033[0m")
    print("\033[94m  -Label: "+str(label))
    print("\033[0m")

    # create the output signals for this label
    targets = numpy.zeros(net.onodes) + 0.01
    targets[label] = 0.99
    # get image data
    image_data = net.backquery(targets)
    __showimage(image_data)

    pass

def guide():
    print("\033[94m\033[1m")
    print("[GUIDE]\033[0m")
    print("\033[94m  -createnet(input_nodes, hidden_nodes, output_nodes, learning_rate, learning_epochs)")
    print("  -trainnet()")
    print("  -testnet()")
    print("  -test(id)")
    print("  -testrandom()")
    print("  -testimage(image_file_name)")
    print("  -showidea(label)")
    print("  -quit()")
    print("\033[0m")

    pass

def __error():
    print("\033[1m\033[91m")
    print("[ERROR]\033[0m")
    print("\033[91m  -You have to create a network first.\n  -Type guide() for help.")
    print("\033[0m")
    pass

def __showimage(all_values):
    # reshape the received data to an image array
    image_array = all_values.reshape((28, 28))
    # show this data in an extra window
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    pass
