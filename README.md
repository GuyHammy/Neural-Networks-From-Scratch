# Fashion-MNIST: Perceptron & Neural Network from Scratch

A project from my postgraduate course *Deep Learning and Neural Networks*, implementing a Perceptron and a
feedforward Artificial Neural Network entirely from scratch (no PyTorch/TensorFlow/sklearn) and using both to
classify [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) images:

- **[Perceptron](./perceptron.ipynb)** — step and sigmoid activations, online and batch learning, binary
  (clothing vs. shoes) then extended to full 10-class classification via one-vs-all
- **[Neural Network](./neural_network.ipynb)** — configurable hidden layers, sigmoid and ReLU activations,
  binary then multi-class classification with hyperparameter tuning

This README summarizes the approach and findings for both. The full implementation, experiments, and reasoning
live in the notebooks themselves.

## Dataset

Fashion-MNIST: 28x28 greyscale images of clothing items across 10 classes (T-shirt/top, trouser, pullover,
dress, coat, sandal, shirt, sneaker, bag, ankle boot), provided as flattened 784-pixel CSV rows with a label
column. The binary tasks collapse this into clothing-vs-shoe; the multi-class tasks use all 10 classes.

## Perceptron

**Binary classification (clothing vs. shoes)**

| Activation | Learning | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Step | Online | **0.9929** | 0.9706 | 0.958 |
| Step | Batch | 0.9829 | 0.9200 | 0.908 |
| Sigmoid | Online | **0.9929** | **0.9813** | 0.947 |
| Sigmoid | Batch | 0.9845 | 0.9316 | 0.912 |

Online learning outperformed batch learning for both activations, since it updates weights after every sample
rather than withholding updates until a full pass over the data.

**Multi-class classification (one-vs-all, 10 classes)**

| Activation | Learning | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **Step** | **Online** | **0.7228** | **0.7475** | **0.7228** | **0.6985** |
| Step | Batch | 0.5677 | 0.6109 | 0.5677 | 0.5242 |
| Sigmoid | Online | 0.6571 | 0.7081 | 0.6571 | 0.6255 |
| Sigmoid | Batch | 0.6504 | 0.5764 | 0.6504 | 0.5974 |

Step activation with online learning was the strongest performer, somewhat surprisingly outperforming sigmoid.
A likely explanation is that sigmoid's smaller, smoother weight updates need more iterations to converge, while
batch updates in general struggled to work well with the strict binary step function.

## Neural Network

**Hyperparameter tuning (binary classification)**

| Hidden layers | Learning rate | Batch size | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| [3, 4, 5] | 0.01 | 64 | 0.9000 | 0.0000 | 0.0000 |
| [3, 4, 5] | 0.2 | 64 | 0.9319 | 0.7813 | 0.4430 |
| [3, 4, 5] | 0.1 | 100 | 0.9864 | 0.9364 | 0.9270 |
| **[30, 40, 50]** | **0.1** | **100** | **0.9863** | **0.9481** | **0.9130** |

A learning rate of 0.01 was too low to converge within the iteration budget, and 0.2 overshot; 0.1 struck the
right balance. Widening the hidden layers further gave marginal returns for the binary task.

**Multi-class classification (10 classes)**

| Hidden layers | Accuracy |
|---|---|
| [3, 4, 5] | 0.3006 |
| [30, 40, 50] | 0.7986 |
| [80, 70, 60] (sigmoid) | 0.8142 |
| **[80, 70, 60] (ReLU)** | **0.8776** |

Multi-class classification needed substantially wider hidden layers than the binary task to have enough
capacity to separate 10 classes, and switching the final configuration from sigmoid to ReLU gave a further
accuracy boost by mitigating vanishing gradients.

**Perceptron vs. neural network (binary, sigmoid)**

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Neural Network | 0.9863 | 0.9481 | 0.9130 |
| **Perceptron** | **0.9929** | **0.9813** | 0.947 |

For this simple binary task, the perceptron matched or exceeded the neural network on every metric while
training much faster — extra model capacity wasn't needed to separate two classes.

**Design choices**

- **Weight initialisation** — Xavier initialisation (scaling a normal distribution by the inverse square root
  of the layer's input size) to keep activations from exploding or vanishing across layers.
- **Iterations** — capped at 20, balancing convergence against training time and overfitting risk.
- **Layers** — 3 hidden layers, sized [80, 70, 60] for the final configuration, chosen as a practical trade-off
  between capacity and the available compute.

## Key takeaways

- Online learning consistently outperformed batch learning across both models, since weight updates propagate
  immediately rather than being averaged away over an epoch.
- Multi-class classification needs meaningfully wider hidden layers than binary classification to have enough
  capacity to separate all classes — [3, 4, 5] nodes was enough for binary but collapsed to ~30% accuracy on 10
  classes.
- ReLU outperformed sigmoid on the multi-class task by mitigating vanishing gradients, but for the simple
  binary task the extra capacity of a neural network (over a single perceptron) wasn't worth its training cost.

## Running it

```bash
pip install numpy matplotlib
jupyter notebook perceptron.ipynb
jupyter notebook neural_network.ipynb
```

Note: the Fashion-MNIST CSVs (`mnist_fashion_train.csv`, `mnist_fashion_test.csv`) were provided as part of the
coursework and aren't redistributed here — see the [Fashion-MNIST repo](https://github.com/zalandoresearch/fashion-mnist)
for the original dataset.
