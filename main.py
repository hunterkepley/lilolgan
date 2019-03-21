import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# get_y and sample_data generates random samples
# then generating the second coordinate using a function
# This is for our true dataset.
def get_y(x):
    return 10 + x**4 # Make this any function (10 + FUNC to move 10 above the axis, obviously)

def sample_Z(m, n):
    return np.random.uniform(-100., 100., size=[m, n])

def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)

# Generator network
# hsize is the number of units in the 2 hidden layers
def generator(Z, hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)

    return out

# Discriminator network
def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)
    
    return out, h3

batch_size = 256
nd_steps = 10
ng_steps = 10

# For plotting the training data
x_plot = sample_data(n=batch_size)

# Adversarial training
X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])

# Create the graph for generating samples from Generator and feeding
# real and generated samples to the Discriminator network.
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

# Define the loss functions for the Generator and Discriminator
# using the logits we just defined
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Set up figure to draw points
fig = plt.gcf()
fig.show()
fig.canvas.draw()

# Train each network alternating
for i in range(100001):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)

    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    # update figure to show training set and generated points
    if i % 100 == 0:
        plt.clf()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        plt.scatter(x_plot[:,0], x_plot[:,1])
        plt.scatter(g_plot[:,0], g_plot[:,1])
        fig.canvas.draw()

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))