# Load required modules.
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

# Setting the working space.
wd = r"C:\Users\Jiseong Yang\git_projects\DSC3006"
wd = wd.replace("'\'", "/")
os.chdir(wd)
os.getcwd()

# Load and split the dataset
xy = np.loadtxt('bank_dataset.csv',delimiter=',',dtype=np.float32)
xy_train,xy_test = train_test_split(xy,test_size=0.2) 

# Scaling of the data
xy_1 = xy_train[:,0:5]
xy_2 = xy_train[:,5:10]
xy_standadized_skl = StandardScaler().fit_transform(xy_1)
xy_train = np.concatenate((xy_standadized_skl,xy_2), axis=1)

xy_3 = xy_test[:,0:5]
xy_4 = xy_test[:,5:10]
xy_standadized_sk2 = StandardScaler().fit_transform(xy_3)
xy_test = np.concatenate((xy_standadized_sk2,xy_4), axis=1)

# Split the target variable.
x_data_train = xy_train[:,:-1]
y_data_train = xy_train[:,[-1]]
x_data_test = xy_test[:,:-1]
y_data_test = xy_test[:,[-1]]

# Set the hyperparameter.
tf.reset_default_graph()
learning_rate = 0.003
keep_prob = tf.placeholder_with_default(0.6, shape=())
threshold = tf.placeholder_with_default(0.65, shape=())

# Placeholders
X = tf.placeholder(tf.float32, [None, 9])
Y = tf.placeholder(tf.float32, [None, 1])

# The number of inputs and outputs.
num_in = 9
num_hidden = 45
num_out = 1
    
# Layer 1
W1 = tf.get_variable("weight1", shape=[num_in,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("bias1", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)
L1 =tf.nn.dropout(_L1, keep_prob)

# Layer 2
W2 = tf.get_variable("weight2", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("bias2", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L2 = tf.nn.leaky_relu(tf.matmul(L1, W2) + b2)
L2 =tf.nn.dropout(_L2, keep_prob)
 
# Layer 3
W3 = tf.get_variable("weight3", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("bias3", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L3 = tf.nn.leaky_relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, keep_prob)

# Layer 4
W4 = tf.get_variable("weight4", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("bias4", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L4 = tf.nn.leaky_relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(_L4, keep_prob)

# Layer 5
W5 = tf.get_variable("weight5", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("bias5", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L5 = tf.nn.leaky_relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(_L5, keep_prob)

# Layer 6
W6 = tf.get_variable("weight6", shape=[num_hidden,num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.get_variable("bias6", shape=[num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
hypothesis = tf.sigmoid(tf.matmul(L5, W6) + b6)
hypothesis = tf.clip_by_value(hypothesis,1e-1, 1-(1e-1))

# Define the cost and hypothesis function.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > threshold, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Define a cost history array for later visualziation.
cost_history = np.empty(shape=[1], dtype=float)

# Model fitting.
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    print("Starts learning...")
    for step in range(1001):
        sess.run(train, feed_dict={X: x_data_train, Y: y_data_train})
        if step % 50 == 0:
            print("step ",step, "cost ", sess.run(cost, feed_dict={
                  X: x_data_train, Y: y_data_train}))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data_test, Y: y_data_test, keep_prob: 1.0})
    
    # Report the result.
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
    