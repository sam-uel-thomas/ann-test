import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load and preprocess data
data = pd.read_csv('../data/mnist_train.csv')
data = np.array(data)
num_samples, num_features = data.shape
np.random.shuffle(data)

# Split data into training and validation sets
data_dev = data[0:1000].T
labels_dev = data_dev[0]
features_dev = data_dev[1:num_features]
features_dev = features_dev / 255.

data_train = data[1000:num_samples].T
labels_train = data_train[0]
features_train = data_train[1:num_features]
features_train = features_train / 255.

# Initialize parameters (weights and biases)
def init_params():
    weights1 = np.random.rand(10, 784) - 0.5
    biases1 = np.random.rand(10, 1) - 0.5
    weights2 = np.random.rand(10, 10) - 0.5
    biases2 = np.random.rand(10, 1) - 0.5
    return weights1, biases1, weights2, biases2

# ReLU activation function
def ReLU(linear_output):
    return np.maximum(linear_output, 0)

# Softmax activation function
def softmax(linear_output):
    exp_values = np.exp(linear_output - np.max(linear_output, axis=0, keepdims=True))
    return exp_values / (np.sum(exp_values, axis=0, keepdims=True) + 1e-9)

# Forward propagation
def forward_prop(weights1, biases1, weights2, biases2, features):
    linear_output1 = weights1.dot(features) + biases1
    activation_output1 = ReLU(linear_output1)
    linear_output2 = weights2.dot(activation_output1) + biases2
    activation_output2 = softmax(linear_output2)
    return linear_output1, activation_output1, linear_output2, activation_output2

# One-hot encode labels
def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels.T

# Derivative of ReLU activation function
def deriv_ReLU(linear_output):
    return linear_output > 0

# Backward propagation
def back_prop(linear_output1, activation_output1, linear_output2, activation_output2, weights2, features, labels):
    m = labels.size
    one_hot_labels = one_hot(labels)
    dLinear_output2 = activation_output2 - one_hot_labels
    dWeights2 = 1 / m * dLinear_output2.dot(activation_output1.T)
    dBiases2 = 1 / m * np.sum(dLinear_output2, axis=1, keepdims=True)
    dLinear_output1 = weights2.T.dot(dLinear_output2) * deriv_ReLU(linear_output1)
    dWeights1 = 1 / m * dLinear_output1.dot(features.T)
    dBiases1 = 1 / m * np.sum(dLinear_output1, axis=1, keepdims=True)
    return dWeights1, dBiases1, dWeights2, dBiases2

# Update parameters (weights and biases)
def update_params(weights1, biases1, weights2, biases2, dWeights1, dBiases1, dWeights2, dBiases2, learning_rate):
    weights1 = weights1 - learning_rate * dWeights1
    biases1 = biases1 - learning_rate * dBiases1
    weights2 = weights2 - learning_rate * dWeights2
    biases2 = biases2 - learning_rate * dBiases2
    return weights1, biases1, weights2, biases2

# Get predictions from output layer
def get_predictions(activation_output2):
    return np.argmax(activation_output2, axis=0)

# Calculate accuracy of predictions
def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

# Perform gradient descent to train the model
def gradient_descent(features, labels, learning_rate, iterations):
    weights1, biases1, weights2, biases2 = init_params()
    for i in range(iterations):
        linear_output1, activation_output1, linear_output2, activation_output2 = forward_prop(weights1, biases1, weights2, biases2, features)
        dWeights1, dBiases1, dWeights2, dBiases2 = back_prop(linear_output1, activation_output1, linear_output2, activation_output2, weights2, features, labels)
        weights1, biases1, weights2, biases2 = update_params(weights1, biases1, weights2, biases2, dWeights1, dBiases1, dWeights2, dBiases2, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(activation_output2)
            print("Accuracy: ", get_accuracy(predictions, labels))
    return weights1, biases1, weights2, biases2

# Make predictions on new data
def make_predictions(features, weights1, biases1, weights2, biases2):
    _, _, _, activation_output2 = forward_prop(weights1, biases1, weights2, biases2, features)
    predictions = get_predictions(activation_output2)
    return predictions

# Test prediction for a specific index
def test_prediction(index, weights1, biases1, weights2, biases2):
    current_image = features_train[:, index, None]
    prediction = make_predictions(features_train[:, index, None], weights1, biases1, weights2, biases2)
    label = labels_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Save parameters to files
def save_params(weights1, biases1, weights2, biases2):
    np.save('../model_params/weights1.npy', weights1)
    np.save('../model_params/biases1.npy', biases1)
    np.save('../model_params/weights2.npy', weights2)
    np.save('../model_params/biases2.npy', biases2)

# Load parameters from files
def load_params():
    weights1 = np.load('../model_params/weights1.npy')
    biases1 = np.load('../model_params/biases1.npy')
    weights2 = np.load('../model_params/weights2.npy')
    biases2 = np.load('../model_params/biases2.npy')
    return weights1, biases1, weights2, biases2

# Check if saved parameters exist
def saved_params_exist():
    return os.path.isfile('../model_params/weights1.npy') and os.path.isfile('../model_params/biases1.npy') and os.path.isfile('../model_params/weights2.npy') and os.path.isfile('../model_params/biases2.npy')

# Main function to either retrain or load parameters
def main():
    if saved_params_exist():
        choice = input("Saved parameters found. Do you want to retrain the model? (yes/no): ").strip().lower()
        if choice == 'no':
            weights1, biases1, weights2, biases2 = load_params()
        else:
            weights1, biases1, weights2, biases2 = retrain_model()
    else:
        weights1, biases1, weights2, biases2 = retrain_model()

    # Make predictions on dev set
    dev_predictions = make_predictions(features_dev, weights1, biases1, weights2, biases2)
    dev_accuracy = get_accuracy(dev_predictions, labels_dev)
    print("Dev set accuracy: ", dev_accuracy)

    # Interactive testing
    while True:
        index = input("Enter an index to test (or 'exit' to quit): ")
        if index.lower() == 'exit':
            break
        try:
            index = int(index)
            if 0 <= index < features_train.shape[1]:
                test_prediction(index, weights1, biases1, weights2, biases2)
            else:
                print("Index out of range. Please enter a valid index.")
        except ValueError:
            print("Invalid input. Please enter a valid index.")

# Function to retrain the model and save parameters if accuracy improves
def retrain_model():
    weights1, biases1, weights2, biases2 = gradient_descent(features_train, labels_train, 0.10, 2000)
    dev_predictions = make_predictions(features_dev, weights1, biases1, weights2, biases2)
    dev_accuracy = get_accuracy(dev_predictions, labels_dev)
    print("New dev set accuracy: ", dev_accuracy)

    if saved_params_exist():
        old_weights1, old_biases1, old_weights2, old_biases2 = load_params()
        old_dev_predictions = make_predictions(features_dev, old_weights1, old_biases1, old_weights2, old_biases2)
        old_dev_accuracy = get_accuracy(old_dev_predictions, labels_dev)
        if dev_accuracy > old_dev_accuracy:
            print("New model is better. Saving new parameters.")
            save_params(weights1, biases1, weights2, biases2)
        else:
            print("Old model is better. Keeping old parameters.")
            weights1, biases1, weights2, biases2 = old_weights1, old_biases1, old_weights2, old_biases2
    else:
        print("Saving new parameters.")
        save_params(weights1, biases1, weights2, biases2)

    return weights1, biases1, weights2, biases2

if __name__ == "__main__":
    main()