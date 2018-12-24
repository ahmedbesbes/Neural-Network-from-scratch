
## Learn backpropagtion the **hard** way

![Backpropagation](https://github.com/ahmedbesbes/Neural-Network-from-scratch/blob/master/images/backprop.gif)

In this repository, I will show you how to build a neural network from scratch (yes, by using plain python code with no framework involved) that trains by mini-batches using gradient descent. Check **nn.py** for the code.

In the related notebook **Neural_Network_from_scratch_with_Numpy.ipynb** we will test nn.py on a set of non-linear classification problems

- We'll train the neural network for some number of epochs and some hyperparameters
- Plot a live/interactive decision boundary 
- Plot the train and validation metrics such as the loss and the accuracies


## Example: Noisy Moons (Check the notebook for other kinds of problems)

### Decision boundary (you'll get to this graph animated during training)
![Decision boundary](https://github.com/ahmedbesbes/Neural-Network-from-scratch/blob/master/images/decision_boundary.png)

### Loss and accuracy monitoring on train and validation sets 
![Loss/Accuracy monitoring on train/val](https://github.com/ahmedbesbes/Neural-Network-from-scratch/blob/master/images/loss_acc.png)


## Where to go from here?
nn.py is a toy neural network that is meant for educational purposes only. So there's room for a lot of improvement if you want to pimp it. Here are some guidelines:

- Implement a different loss function such as the Binary Cross Entropy loss. For a classification problem, this loss works better than a Mean Square Error. 
- Make the code generic regarding the activation functions so that we can choose any function we want: ReLU, Sigmoid, Tanh, etc.
- Try to code another optimizers: SGD is good but it has some limitations: sometimes it can be stuck in local minima. Look into Adam or RMSProp.
- Play with the hyperparameters and check the validation metrics

