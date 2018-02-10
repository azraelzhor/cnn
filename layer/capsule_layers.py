import tensorflow as tf

def squash(input):
    length_square = tf.reduce_sum(tf.square(input), axis=2)
    length = tf.sqrt(length_square)

    length = tf.expand_dims(length, -1)
    length_square = tf.expand_dims(length_square, -1)

    norm = length / (1 + length_square)
    out = input * norm

    return out

def digit_cap(input, r):

    # input: D x 1152 x 8
    shape = input.get_shape()

    # weight: 1152 x 10 x 8 x 16
    weight = tf.get_variable("weights", [shape[1], 10, shape[2], 16], trainable=True)

    # coupling coefficients: 1152 x 10
    c = tf.get_variable("coupling_coefs", [shape[1], 10], trainable=False)

    # logit init
    b = tf.zeros([shape[1], 10])

    input_t = tf.transpose(input, perm=[1, 0, 2])
    input_t = tf.reshape(tf.tile(input_t, [10, 1, 1]), [tf.shape(input)[1], 10, -1, 8])

    for i in range(r):
        c = tf.nn.softmax(b, dim=1)

        c_reshaped = tf.reshape(c, [tf.shape(input)[1], 10, 1, 1])

        pred = tf.matmul(input_t, weight)
        
        s = tf.transpose(tf.reduce_sum(pred * c_reshaped, axis=0), perm=[1, 0, 2])

        v = squash(s)

        pred_t = tf.transpose(pred, perm=[2, 1, 0, 3])
        v_e = tf.expand_dims(v, -1)
        delta = tf.squeeze(tf.reduce_sum(tf.matmul(pred_t, v_e), axis=0), axis=2)
        b = b + tf.transpose(delta)

    return v

def primary_cap(input):
    out = tf.layers.conv2d(inputs=input, filters=256, kernel_size=[9, 9], strides=(2, 2))
    out_reshaped = tf.reshape(out, [tf.shape(out)[0], 1152, 8])
    
    return out_reshaped

def debug():

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])

    conv_1 = tf.layers.conv2d(inputs=X, filters=256, kernel_size=[9, 9], strides=(1, 1), activation=tf.nn.relu)

    pri_cap = primary_cap(conv_1)

    di_cap = digit_cap(pri_cap, 1) 

    print("X: ", X.get_shape())
    print("conv1: ", conv_1.get_shape())
    print("pricap: ", pri_cap.get_shape())
    print("dicap: ", di_cap.get_shape())

if __name__ == "__main__":
    
    debug()

    # debug
    t = tf.constant([[[1, 1, 1],
                     [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    m = tf.constant([[1, 2, 5], [8, 4, 9]], dtype=tf.float32)
    x = tf.Variable(0)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(tf.nn.softmax(m, dim=1))
    print(result)
