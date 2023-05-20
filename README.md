# Neural Network with Backpropagation Algorithm for XOR Problem

This C++ program implements a neural network with backpropagation algorithm to solve the XOR problem. The XOR problem is a classic problem in neural network literature, where the goal is to predict the output of an XOR gate given two binary inputs.

## Dependencies

This program requires the following dependencies:

- C++ compiler that supports C++11 or later
- Standard Template Library (STL)
- <vector> library
- <iostream> library
- <cmath> library
- <ctime> library

## How to Run

To compile and run the program, navigate to the directory containing the source code and type the following commands in the terminal:

```
$ g++ -std=c++11 neural_network.cpp -o neural_network
$ ./neural_network
```

The program will output the predicted output for all possible input vectors in the XOR problem, as well as the error rate and number of iterations.

## Program Structure

The neural network is implemented in the `NeuralNetwork` class in the `neural_network.cpp` file. The class has the following public methods:

- `NeuralNetwork(int numInputs, int numOutputs, vector<int> hiddenLayerSizes)`: Constructor that initializes the neural network with the specified number of inputs, outputs, and hidden layer sizes.
- `void feedForward(vector<double> inputs)`: Feeds the input vector forward through the neural network to compute the output.
- `void backpropagate(vector<double> inputs, vector<double> expectedOutputs, double learningRate)`: Adjusts the weights of the neurons using backpropagation algorithm to minimize the error between the predicted output and the expected output.
- `double predict(vector<double> inputs)`: Predicts the output for the input vector using the trained neural network.

The main function in `neural_network.cpp` demonstrates how to use the `NeuralNetwork` class to solve the XOR problem.

## License

This program is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
