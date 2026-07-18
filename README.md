# LearnPlusPlus

LearnPlusPlus is a C++ deep learning library that provides neural networks capable of performing regression and classification tasks.

The user has complete freedom over model architecture (input size, hidden layers, activations). Model construction, training, and inference are all wrapped in a user-friendly interface through the ```LPP::Network``` class. After training, model architecture and weights may be exported.

LearnPlusPlus was developed with latency and memory safety in mind:
- Mutlithreaded matrix operations enable lightning fast inference
- Preallocated buffers ensure minimal memory allocations in the training hotpath
- All memory management is handled using smart pointers, meaning LearnPlusPlus will **never** introduce memory leaks

## How to Use?

Initalize model with your own chosen architecture:
```cpp
LPP::Network my_model(
  10,  // Input size
  {
    {32, &LPP::activations::relu,    nullptr}, // nullptr means weights are initialized using He initialization
    {16, &LPP::activations::relu,    nullptr},
    {8,  &LPP::activations::relu,    nullptr},
    {7,  &LPP::activations::softmax, nullptr}
  });
```

Load your data:
```cpp
std::vector<std::vector<float>> data1 = ...;
std::vector<std::vector<float>> data2 = ...;

auto X = LPP::Matrix(data1); 
auto Y = LPP::Matrix(data2);
```

Set basic training options and train:
```cpp
size_t epochs       = 10;
float learning_rate = 0.001;
auto loss_func      = &LPP::losses::cross_entropy;

my_model.train(X, Y, epochs, learning_rate, loss_func);
```

Perform inference:
```cpp
std::vector<float> x      = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
std::vector<float> result = my_model.inference(x);
```

## Optimizers

Add optimizers/advanced training options:
```cpp
// Create training config
auto config = LPP::ExtraTrainingOptions();

// Minibatches
config.set_mini_batch_size(32);
// Regularization
auto L2 = &LPP::regular::ridge;
config.set_regularization(L2);
L2->set_lambda(1e-04f);
// Validation
config.set_validation_data(X_val, Y_val);

// Train
my_model.train(X, Y, epochs, learning_rate, loss_func, config);
```

## Model Serialization

You can also save your weights and load other models, which could be further trained.
```cpp
// Save
my_model.save_model("filename.txt");
// Load
LPP::Network loaded_model("filename.txt");
```

For a full example, see the examples folder!
