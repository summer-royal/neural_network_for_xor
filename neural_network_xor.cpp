#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

const double e = 2.71828182845904523536;

class Neuron {
public:
    double output;
    vector<double> weights;
    vector<double> lastDelta;
    Neuron() {}
    Neuron(int numInputs) {
        for(int i = 0; i < numInputs; i++) {
            weights.push_back(getRandomWeight());  // Initialize random weights for each input connection
            lastDelta.push_back(0);  // Initialize lastDelta values for weight update momentum
        }
    }
    double getRandomWeight() {
        return (double)rand() / (double)RAND_MAX;  // Generate random weight between 0 and 1
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    Layer() {}
    Layer(int numNeurons, int numInputsPerNeuron) {
        for(int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputsPerNeuron));  // Create neurons with specified number of inputs
        }
    }
};

class NeuralNetwork {
public:
    int numInputs;
    int numOutputs;
    vector<Layer> hiddenLayers;
    NeuralNetwork(int numInputs, int numOutputs, vector<int> numNeuronsInHiddenLayers) {
        this->numInputs = numInputs;
        this->numOutputs = numOutputs;
        for(int i = 0; i < numNeuronsInHiddenLayers.size(); i++) {
            if(i == 0) {
                hiddenLayers.push_back(Layer(numNeuronsInHiddenLayers[i], numInputs));  // First hidden layer receives inputs
            } else {
                hiddenLayers.push_back(Layer(numNeuronsInHiddenLayers[i], numNeuronsInHiddenLayers[i - 1]));  // Subsequent layers receive outputs from previous layer
            }
        }
        hiddenLayers.push_back(Layer(numOutputs, numNeuronsInHiddenLayers[numNeuronsInHiddenLayers.size() - 1]));  // Output layer
    }
    double sigmoid(double x) {
        return 1.0 / (1.0 + pow(e, -x));  // Sigmoid activation function
    }
    double sigmoidDerivative(double x) {
        return sigmoid(x) * (1.0 - sigmoid(x));  // Derivative of the sigmoid function
    }
    void feedForward(vector<double> inputs) {
        for(int i = 0; i < hiddenLayers[0].neurons.size(); i++) {
            double sum = 0;
            for(int j = 0; j < inputs.size(); j++) {
                sum += inputs[j] * hiddenLayers[0].neurons[i].weights[j];  // Weighted sum of inputs multiplied by respective weights
            }
            hiddenLayers[0].neurons[i].output = sigmoid(sum);  // Apply activation function and store output in neuron
        }
        for(int i = 1; i < hiddenLayers.size(); i++) {
            for(int j = 0; j < hiddenLayers[i].neurons.size(); j++) {
                double sum = 0;
                for(int k = 0; k < hiddenLayers[i - 1].neurons.size(); k++) {
                    sum += hiddenLayers[i - 1].neurons[k].output * hiddenLayers[i].neurons[j].weights[k];  // Weighted sum of outputs from previous layer multiplied by respective weights
                }
                hiddenLayers[i].neurons[j].output = sigmoid(sum);  // Apply activation function and store output in neuron
            }
        }
    }
    void backpropagate(vector<double> inputs, vector<double> expectedOutputs, double learningRate) {
        vector<double> errors;
        for(int i = 0; i < hiddenLayers[hiddenLayers.size() - 1].neurons.size(); i++) {
            double error = expectedOutputs[i] - hiddenLayers[hiddenLayers.size() - 1].neurons[i].output;  // Calculate error for each output neuron
            errors.push_back(error * sigmoidDerivative(hiddenLayers[hiddenLayers.size() - 1].neurons[i].output));  // Multiply error by derivative of sigmoid for gradient descent
            for(int j = 0; j < hiddenLayers[hiddenLayers.size() - 2].neurons.size(); j++) {
                double delta = learningRate * errors[i] * hiddenLayers[hiddenLayers.size() - 2].neurons[j].output;  // Calculate weight update delta
                hiddenLayers[hiddenLayers.size() - 1].neurons[i].weights[j] += delta + hiddenLayers[hiddenLayers.size() - 1].neurons[i].lastDelta[j] * 0.1;  // Apply weight update with momentum
                hiddenLayers[hiddenLayers.size() - 1].neurons[i].lastDelta[j] = delta;  // Store delta for momentum calculation in next iteration
            }
        }
        for(int i = hiddenLayers.size() - 2; i >= 0; i--) {
            vector<double> nextErrors;
            for(int j = 0; j < hiddenLayers[i].neurons.size(); j++) {
                double error = 0;
                for(int k = 0; k < hiddenLayers[i + 1].neurons.size(); k++) {
                    error += errors[k] * hiddenLayers[i + 1].neurons[k].weights[j];  // Calculate error for each neuron in hidden layers
                }
                nextErrors.push_back(error * sigmoidDerivative(hiddenLayers[i].neurons[j].output));  // Multiply error by derivative of sigmoid for gradient descent
                for(int k = 0; k < hiddenLayers[i].neurons[j].weights.size(); k++) {
                    double delta = learningRate * nextErrors[j] * (i == 0 ? inputs[k] : hiddenLayers[i - 1].neurons[k].output);  // Calculate weight update delta
                    hiddenLayers[i].neurons[j].weights[k] += delta + hiddenLayers[i].neurons[j].lastDelta[k] * 0.1;  // Apply weight update with momentum
                    hiddenLayers[i].neurons[j].lastDelta[k] = delta;  // Store delta for momentum calculation in next iteration
                }
            }
            errors = nextErrors;
        }
    }
    double predict(vector<double> inputs) {
        feedForward(inputs);  // Feed inputs forward through the neural network
        return hiddenLayers[hiddenLayers.size() - 1].neurons[0].output;  // Return the output of the last neuron in the output layer
    }
};

int main() {
    srand(time(NULL));  // Seed the random number generator
    NeuralNetwork nn(2, 1, { 2 });  // Create a neural network with 2 inputs, 1 output, and a hidden layer of 2 neurons
    vector<vector<double>> inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };  // XOR input vectors
    vector<double> expectedOutputs = { 0, 1, 1, 0 };  // Expected outputs for the XOR problem
    int numIterations = 10000;  // Number of training iterations
    double learningRate = 0.5;  // Learning rate for weight updates
    for(int i = 0; i < numIterations; i++) {
        int j = rand() % inputs.size();  // Randomly select an input vector for training
        nn.feedForward(inputs[j]);  // Feed the input vector forward through the neural network
        nn.backpropagate(inputs[j], { expectedOutputs[j] }, learningRate);  // Backpropagate to adjust weights based on the error
    }
    for(int i = 0; i < inputs.size(); i++) {
        cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << nn.predict(inputs[i]) << endl;  // Predict the output for each input vector in the XOR problem
    }
    return 0;
}
