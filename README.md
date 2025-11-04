# LearnPlusPlus

LearnPlusPlus is a C++ deep learning library that provides neural networks capable of performing regression and classification tasks. The user has complete freedom over model architecture (input size, hidden layers, activations) -- they can specify their own or load it in from a file. Model construction, training, and inference are all wrapped in a user-friendly interface through the ```LPP::Network``` class. Mutlithreaded vector and matrix operations have also been implemented, allowing for lightning fast inference. After training, weights can also be exported to allow for transfer learning or easy inference by other programs. Finally, all memory management is handled using smart pointers, meaning that LearnPlusPlus will **never** introduce memory leaks into your program.

## How to Use?

Initalize model with your own chosen architecture:
```cpp
LPP::Network my_model(
  10,                    // Input size of 10
  {
    {32, LPP::RELU},          // Hidden layer with 32 nodes, ReLU activation
    {16, LPP::RELU},          // Hidden layer with 16 nodes, ReLU activation
    {8,  LPP::RELU},          // Hidden layer with 8 nodes,  ReLU activation
    {4,  LPP::SIGMOID}        // Output layer with 4 values (classification use case)
  },
  LPP::STANDARD_NORMAL   // Distribution which initial weights are sampled from
);
```

Load your data:
```cpp
std::vector<std::vector<double>> data1 = ...;
std::vector<std::vector<double>> data2 = ...;

const LPP::Matrix explanatory_variates(data1);    // Construction can also be done in place
const LPP::Matrix response_variates(data2);
```

Set number of epochs, learning rate, and loss function for training:
```cpp
const size_t epochs        = 10;
const double learning_rate = 0.0001;
const auto loss_func       = LPP::MEAN_SQUARED_ERROR;
```

Call train!
```cpp
my_model.train(explanatory_variates, response_variates, epochs, learning_rate, loss_func);
```

Perform inference and see its results:
```cpp
const std::vector<double> x      = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const std::vector<double> result = my_model.inference(x);

LPP::print_object(result);
```

Save your model:
```cpp
my_model.save_model("filename.txt");
```

Initialize model from file:
```cpp
LPP::Network loaded_model("filename.txt");
// Perform inference
// Or continue training
```

For a full example, see ```examples/basic.cpp```, where we train a regression network to learn a simple mathematical function.
