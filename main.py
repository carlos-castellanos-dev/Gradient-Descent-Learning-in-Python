import matplotlib.pyplot as plt
import numpy as np


# Function Definitions

# Plotting Cost Function
def plot_cost_func(J, iterations):
    # Plotting the learning curve
    x = np.arange(iterations, dtype=int)
    y = J
    plt.plot(x, y)
    plt.axis([-1, x.shape[0] + 1, -1, np.max(y) + 1])
    plt.title('Learning Curve')
    plt.xlabel('x: Iteration Number')
    plt.ylabel('y: J(0)')
    plt.show()


# Datapoint Plotting
def plot_fun(features, labels, classes):
    plt.plot(features[labels[:] == classes[0], 0], features[labels[:] == classes[0], 1], 'rs',
             features[labels[:] == classes[1], 0], features[labels[:] == classes[1], 1], 'g^')
    plt.axis([-4, 4, -4, 4])
    plt.xlabel('x: feature 1')
    plt.ylabel('y: feature 2')
    plt.legend(['Class' + str(classes[0]), 'Class' + str(classes[1])])
    plt.show()


# Threshold Plotting
def plot_fun_thr(features, labels, thre_parms, classes):
    plt.plot(features[labels[:] == classes[0], 0], features[labels[:] == classes[0], 1], 'rs',
             features[labels[:] == classes[1], 0], features[labels[:] == classes[1], 1], 'g^')
    plt.axis([-4, 4, -4, 4])
    x1 = np.linspace(-2, 2, 50)
    x2 = -(thre_parms[1] * x1 + thre_parms[0]) / thre_parms[2]
    plt.plot(x1, x2, '-r')
    plt.xlabel('x: feature 1')
    plt.ylabel('y: feature 2')
    plt.legend(['Class' + str(classes[0]), 'Class' + str(classes[1])])
    plt.show()


# Class Definition
class NeuralNetwork(object):

    def __init__(self, learning_r):
        # np.random.seed(1) # Keeps same weight
        self.weight_matrix = 2 * np.random.random((3, 1)) - 1
        self.l_rate = learning_r
        self.history = np.array([])

    def hard_limiter(self, x):
        outs = np.zeros(x.shape)
        outs[x > 0] = 1
        return outs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, inputs):
        outs = np.dot(inputs, self.weight_matrix)
        return self.sigmoid(outs)

    def train(self, train_inputs, train_outputs, num_train_iterations=10):
        N = train_inputs.shape[0]
        cost_func = np.array([])
        # Number of iterations we want to perform for this set of input
        for iteration in range(num_train_iterations):
            outputs = self.forward_propagation(train_inputs)
            # Calculate the Error in the output
            error = train_outputs - outputs
            adjustment = (self.l_rate / N) * np.sum(np.multiply(error, train_inputs), axis=0)
            # Calculate the Cost Function
            cost_func = np.append(cost_func, (1 / 2 * N) * np.sum(np.power(error, 2)))
            # Adjust Weight Matrix
            self.weight_matrix[:, 0] += adjustment
            # History Variable
            self.history = np.append(self.history, iteration)
            self.history = np.append(self.history, self.weight_matrix)
            self.history = np.append(self.history, (1 / 2 * N) * np.sum(np.power(error, 2)))
            # Plot the seperating line based on the weights
            if (iteration % 5 == 0):
                print('Iteration #' + str(iteration))
                plot_fun_thr(train_inputs[:, 1:3], train_outputs[:, 0], self.weight_matrix[:, 0], classes)
        print('Final Classifier Line')
        plot_fun_thr(train_inputs[:, 1:3], train_outputs[:, 0], self.weight_matrix[:, 0], classes)
        plot_cost_func(cost_func, num_train_iterations)
        self.history = self.history.reshape(num_train_iterations, 5)

    def pred(self, inputs):
        preds = self.forward_propagation(inputs)
        return preds


# Main
features = np.array([[1, 1], [1, 0], [0, 1], [-1, -1], [0.5, 3], [0.7, 2], [-1, 0], [-1, 1], [2, 0], [-2, -1]])
print('Inputs')
print(features)
labels = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, -1])
classes = [-1, 1]
print('Data Points Plotted')
plot_fun(features, labels, classes)
# Adding Bias
bias = np.ones((features.shape[0], 1))
print('Bias added to Input Array')
features = np.append(bias, features, axis=1)
print(features)
print('Shape of Bias/Input Array Combined')
print(features.shape)
# Training
learning_r = [1, 0.5, 0.05]
for num_rates in range(3):
    neural_network = NeuralNetwork(learning_r[num_rates])
    print('\n======Starting Training======')
    print('Learning Rate: ', neural_network.l_rate)
    print('Random weights at the start of training')
    print(neural_network.weight_matrix)
    neural_network.train(features, np.expand_dims(labels, axis=1), 50)
    print('New weights after training')
    print(neural_network.weight_matrix)
    print('=============================')
    print('History Variable(Epoch/Weight/Cost): ')
    print('Learning Rate: ', neural_network.l_rate)
    print('  Epoch    W1      W2      W3     Cost ')
    np.set_printoptions(suppress=True)
    print(neural_network.history)
