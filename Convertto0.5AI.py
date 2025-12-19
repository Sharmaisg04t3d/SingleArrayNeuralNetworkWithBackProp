#importing necessary libraries
import random as rand
import math

# Math functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)


# Neuron class
class Neuron:
    # Initializing neuron with random weights and bias
    def __init__(self,num_inputs):
        self.weight = [rand.random() for i in range(num_inputs)]
        self.bias = rand.random()
        print("New Neuron created with weights:", self.weight, "and bias:", self.bias)

    # Activation function
    def activation(self, x):
        y=0
        for i in range(len(self.weight)):
            y += x[i] * self.weight[i]
        y+=self.bias
        self.last_output = sigmoid(y)
        self.last_input = x
        return self.last_output
    
    #Neuron Level Backpropagation
    def backpropagation(self, learning_rate, target):
        gradient = sigmoid_derivative(self.last_output)*2*(self.last_output - target)
        for i in range(len(self.weight)):
            self.weight[i] -= learning_rate * gradient* self.last_input[i]
        self.bias -= learning_rate * gradient
        return self.last_output

# Layer class
class Layer:
    # Initializing layer with given number of neurons
    def __init__(self, num_neurons,num_inputs):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(num_inputs))
        print("New Layer created with", num_neurons, "neurons.")

    # Layer output computation
    def layer_output(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.activation(inputs))
        return outputs
    
    # Layer Level Backpropagation
    def Layer_back_propagation(self, learning_rate, gradients):
        prev_layer_outputs = [0.0] * len(self.neurons[0].last_input)
        
        for j in range(len(self.neurons)):
            neuron = self.neurons[j]
            neuron.backpropagation(learning_rate,neuron.last_output - gradients[j])
            
            for i in range(len(neuron.weight)):
                prev_layer_outputs[i] += gradients[j] * neuron.weight[i]
        return prev_layer_outputs

# Network class
class Network:
    # Initializing network with given layer sizes
    def __init__(self, layer_sizes,num_inputs):
        self.layers = []
        for i in range(len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i],layer_sizes[i-1] if i > 0 else num_inputs))
        print("New Network created with", len(self.layers), "layers.")
    
    # Network output computation
    def network_output(self, inputs):
        for layer in self.layers:
            inputs = layer.layer_output(inputs)
        return inputs
    
    # Network Level Backpropagation
    def Network_back_propagation(self, lr, targets):
        last_layer = self.layers[-1]
        deltas = []
        # Calculate initial deltas for output layer
        for j in range(len(last_layer.neurons)):
            out = last_layer.neurons[j].last_output
            delta = 2 * (out - targets[j]) * sigmoid_derivative(out)
            deltas.append(delta)
        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            deltas = self.layers[i].Layer_back_propagation(lr, deltas)

# Example usage
inputs = [67, 68, 69]
targets = [6.7, 6.7, 6.7]
# Creating a network with 3 hidden layers of 5 neurons each and an output layer of 3 neurons
net = Network([5, 5, 5, 3], 3)
output = net.network_output(inputs)
loss = 0
for i in range(len(output)):
    loss += (output[i] - targets[i]) ** 2
loss /= len(output)
step = 0
print("Initial loss:", loss)
# Training loop
while loss > 0.000000001:
    output = net.network_output(inputs)
    # Calculating loss
    loss = 0
    for i in range(len(output)):
        loss += (output[i] - targets[i]) ** 2
    loss /= len(output)
    print("Current loss:", loss)
    step += 1
    if step % 50 == 0:
        print("Step:", step)
    # Backpropagation
    net.Network_back_propagation(0.1, targets)
print("Training completed in", step, "steps.")
print("Final loss after training:", loss)
print("Final output after training:", net.network_output(inputs))

input1 = float(input("Please enter a number"))
input2 = float(input("Please enter a number"))
input3 = float(input("Please enter a number"))

output = net.network_output([input1, input2, input3])
print("Network output for inputs", [input1, input2, input3], "is:", output)
