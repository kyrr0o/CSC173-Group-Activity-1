# CSC173: NEURAL NETWORK

**Group Members:**
1. Kaycee T. Nalzaro
2. 

## INTRODUCTION
Artificial Neural Network (ANN) is a computational model inspired by the human brain that uses interconnected nodes to process information and make decisions. This group activity aims to build a simple ANN from scratch using only **Python**,**Numpy** and **Matplotlib** for plotting, without any machine-learning libraries. The purpose of this activity is to understand how neural networks work, including forward propagation, loss computation, and backpropagation with gradient descent. The model will be used to perform **binary classification** on the **Breast Cancer Diagnostic Dataset** by using only two features.

## DATA PREPARATION
During the data preparation, we ontained our dataset using the built-in breast cancer diagnostic dataset from scikit-learn:
[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

```python
data = load_breast_cancer()

# Select first two features
X = data.data[:, :2]

y = data.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
The dataset contains measurements of cell nuclei from breast cancer biopsies, with targets labeled as 0 = malignant or 1 = benign. We only selected the first two features which will contain the inout layer with two neurons. We used **StandardScaler**, a data processing tool to normalize the features and to ensure faster and more stable training. We then split the dataset following the standard data proportion rule, training (80%) and testing (20%) sets.

## NETWORK DESIGN
The neural netwrok structure consists three layers with each having different numbers of neurons and activation fucntions. **Sigmoid Function** is an activation function that is used in neural networks to activate neurons. 

```python
input_neurons = 2
hidden_neurons = 3   
output_neurons = 1
lr = 0.1
```
The input layer receives two features from the dataset. The hidden layer, consisting of three neurons, processes these inputs using the sigmoid activation, and the output layer with a single neuron produces the final prediction. A learning rate (lr) of 0.1 controls how fast weights are updated during training.

```python
np.random.seed(42)
W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))
```
Weights were initialized randomly, while biases were initialized to zero.

## TRAINING 
The neural network learns from the training data by repeatedly adjusting its weights and biases to minimize error.

```python
epochs = 800
losses = []

for i in range(epochs):
```
The network was trained for 800 epochs, where each epoch represents one complete pass of the dataset through the model. During each epoch, the network performs two key steps:

1. **Forward Propagation**
   In this step, the data flows forward through the network to produce predictions.

   ```python
   # ompute hidden layer weighted sums
   z1 = np.dot(X_train, W1) + b1
   ```
   The input features are multiplied by the weights and added to the biases, giving the total input for each hidden neuron.

   ```python
   # Applied the Sigmoid Function
   a1 = sigmoid(z1)
   ```
   The sigmoid activation introduces non-linearity, helping the network learn complex relationships.

    ```pyhton
    #  Compute output layer weighted sums and activation
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    ```
    The hidden layer output is passed to the output layer to produce the model’s final prediction.

   ```python
   # Calculate Loss
   loss = mse_loss(y_train, y_pred)
   losses.append(loss)
   ```
   The Mean Squared Error (MSE) function measures the difference between predicted and actual outputs. The loss values are stored for each epoch to track performance.
   
3. **Backpropagation**   
    After calculating the loss, the model performs backpropagation, a process that adjusts weights and biases by propagating the error backward through the network.
    
    ```python
    # Compute gradients for the output layer
    d_loss = y_pred - y_train
    d_z2 = d_loss * sigmoid_derivative(y_pred)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    ```
    These lines calculate how much the output layer contributed to the total error. The gradients (d_W2, d_b2) determine how much to adjust the output weights and biases.
    
    ```python
    # Compute gradients for the hidden layer
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X_train.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    ```
    The gradients (d_W1, d_b1) represent the adjustments needed for the input-to-hidden layer weights and biases.
    
    ```python
    # Update weights and biases
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    ```
    The weights and biases are updated using the calculated gradients scaled by the learning rate (lr). This step enables the network to gradually minimize the loss over time.



















































## BACKPROPAGATION

