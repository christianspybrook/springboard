import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1)
Y_training = training_data_df[['total_earnings']]

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1)
Y_testing = test_data_df[['total_earnings']]

# All data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
# Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Convert dat to matrices
X_train_matrix = X_scaled_training.tolist()
Y_train_matrix = Y_scaled_training.tolist()

X_test_matrix = X_scaled_testing.tolist()
Y_test_matrix = Y_scaled_testing.tolist()

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One:
# Define the layers of the neural network itself

# instantiate initializer
initializer = tf.initializers.GlorotUniform()

# create weight dict
weights = {
    'hl1': tf.Variable(initializer([number_of_inputs, layer_1_nodes])),
    'hl2': tf.Variable(initializer([layer_1_nodes, layer_2_nodes])),
    'hl3': tf.Variable(initializer([layer_2_nodes, layer_3_nodes])),
    'hl_out': tf.Variable(initializer([layer_3_nodes, number_of_outputs]))
}
# create biases dict
biases = {
    'b1': tf.Variable(tf.zeros([layer_1_nodes])),
    'b2': tf.Variable(tf.zeros([layer_2_nodes])),
    'b3': tf.Variable(tf.zeros([layer_3_nodes])),
    'b_out': tf.Variable(tf.zeros([number_of_outputs]))
}


def neural_net(input_data):
    """Builds neural net and returns prediction"""
    layer_1_output = tf.add(tf.matmul(input_data, weights['hl1']), biases['b1'])
    layer_1_output = tf.nn.relu(layer_1_output)
    layer_2_output = tf.add(tf.matmul(layer_1_output, weights['hl2']), biases['b2'])
    layer_2_output = tf.nn.relu(layer_2_output)
    layer_3_output = tf.add(tf.matmul(layer_2_output, weights['hl3']), biases['b3'])
    layer_3_output = tf.nn.relu(layer_3_output)
    prediction = tf.add(tf.matmul(layer_3_output, weights['hl_out']), biases['b_out'])
    return tf.nn.relu(prediction)


# Section Two:
# Define the cost function of the neural network that will measure prediction accuracy during training

def cost(prediction, Y):
    """Returns mean squared error"""
    return tf.reduce_mean(tf.math.squared_difference(prediction, Y))


# Section Three:
# Define the optimizer function that will be run to optimize the neural network

optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a summary operation to log the progress of the network

# Create log file writers to record training progress.
# We'll store training and testing log data separately.
training_writer = tf.summary.create_file_writer('./logs/training')
testing_writer = tf.summary.create_file_writer('./logs/testing')

# create empty list for train loss
train_loss = []


def run_optimization(x, y):
    """Runs training session"""
    with tf.GradientTape() as g:
        # call neural net to get predictions
        pred = neural_net(x)
        # calculate loss of training run
        loss = cost(pred, y)
        train_loss.append(loss)
        # Variables to update, i.e. trainable variables.
        trainable_variables = list(weights.values()) + list(biases.values())
        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# create empty lists for test loss and prediction
test_loss = []


def run_test(x, y):
    """Predicts on test data"""
    pred = neural_net(x)
    loss = cost(pred, y)
    test_loss.append(loss)
    # Unscale the data back to it's original units (dollars)
    global y_predicted_dollars
    y_predicted_dollars = Y_scaler.inverse_transform(pred)[0][0]


# Run training for the given number of steps.
for epoch in range(training_epochs):
    # Run the optimization to update W and b values.
    run_optimization(X_train_matrix, Y_train_matrix)
    # run summary operation to log the progress of the network on training data
    epoch_train_loss = np.mean(train_loss)
    with training_writer.as_default():
        tf.summary.scalar('current_loss', epoch_train_loss, step=epoch)
        # Write the current training status to the log files (Which we can view with TensorBoard)
        tf.summary.flush(writer=training_writer)
    # run test data
    run_test(X_test_matrix, Y_test_matrix)
    # run summary operation to log the progress of the network on test data
    epoch_test_loss = np.mean(test_loss)
    with testing_writer.as_default():
        tf.summary.scalar('current_loss', epoch_test_loss, step=epoch)
        # Write the current test status to the log files
        tf.summary.flush(writer=testing_writer)
    # Print the current training status to the screen
    print("Training epoch: %i, Training loss: %f, Testing loss: %f" % (
        epoch + 1, train_loss[epoch], test_loss[epoch]))

# Training is now complete!
print("Training is complete!")
print("Final Training loss: {}".format(train_loss[training_epochs - 1]))
print("Final Testing loss: {}".format(test_loss[training_epochs - 1]))

# Now that the neural network is trained, let's use it to make predictions for our test data.

real_earnings = test_data_df['total_earnings'].values[0]

print("The actual earnings of Game #1 were $%.2f" % real_earnings)
print("Our neural network predicted earnings of $%.2f" % y_predicted_dollars)
