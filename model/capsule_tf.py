import tensorflow as tf

def cap_net(X, r):
    conv_1 = tf.layers.conv2d(inputs=X, filters=256, kernel_size=[9, 9], strides=(1, 1), activation=tf.nn.relu)
    pri_cap = primary_cap(conv_1)
    Z = digit_cap(pri_cap, r)
    return Z

def compute_cost(Z, y, m_plus, m_minus):

    # D x 10
    z_length = tf.sqrt(tf.reduce_sum(tf.square(Z), axis=2))
    y_one_hot = tf.one_hot(y, 10)

    pos_loss = tf.square(tf.nn.relu(m_plus - z_length))
    neg_loss = tf.square(tf.nn.relu(z_length - m_minus))

    cost = tf.reduce_sum(y_one_hot * pos_loss + 0.5 * (1 - y_one_hot) * neg_loss)

    avg_cost = tf.reduce_mean(cost)

    return avg_cost

def model(X_train, y_train, X_val, y_val, X_test, y_test,
    leaning_rate=0.001, mini_batch=64, num_epochs=50):

    # placeholders
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.int32, [None])

    # output of capsule net
    Z = cap_net(X)

    # cost of capsule net
    cost = compute_cost(Z, y)

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:

        for epoch in range(num_epochs):
            pass
    
if __name__ == "__main__":
    pass