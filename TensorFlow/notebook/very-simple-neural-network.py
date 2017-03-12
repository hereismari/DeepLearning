import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------- plt Helper ------------------

def plot(X):
    plt.figure(figsize=(4,4))
    plt.scatter(X[0], X[1])
    plt.show()

# --------------- Setting up Data  -----------------

# number of examples
N = 50

# Creating a perfect line from [-2, -6] to [4, 6]
X =  np.array([np.linspace(-2, 4, N), np.linspace(-6, 6, N)])
plot(X)

# Adding random "noise"
X += np.random.randn(2, N)
plot(X)

# Splitting into x and y
x, y = X

# Add the bias node with value 1
x_bias = np.array([(1., a) for a in x]).astype(np.float32)

# keep track of the errors
errors = []

# number of iterations for gradient descent
GD = 50

# alhpa: learning rate, step size for gradient descent
alpha = 0.002

with tf.Session() as sess:
    # input
    inp = tf.constant(x_bias)

    # target
    target = tf.constant(np.transpose([y]).astype(np.float32))

    # weights. It will change in every iteraction.
    # It starts as random values.
    weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

    # starting all variables above
    tf.global_variables_initializer().run()


    # y = w2 * x + w1
    yhat = tf.matmul(inp, weights)

    # error
    yerror = tf.subtract(yhat, target)

    # calculating L2 loss
    loss = tf.nn.l2_loss(yerror)

    # Gradient descent
    # will update the weights
    gd = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # RUN BABY
    for _ in xrange(GD):
        sess.run(gd)
        errors.append(loss.eval())

    betas = weights.eval()
    yhat = yhat.eval()


# Show results
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.3)
fig.set_size_inches(10, 4)
ax1.scatter(x, y, alpha=.7)
ax1.scatter(x, np.transpose(yhat)[0], c="g", alpha=.6)
line_x_range = (-4, 6)
ax1.plot(line_x_range, [betas[0] + a * betas[1] for a in line_x_range], "g", alpha=0.6)
ax2.plot(range(0, GD), errors)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Training steps")
plt.show()

