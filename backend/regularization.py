
import tensorflow as tf


# regularizations
def regularization(model):
    reg = 0

    # L1 weight regularization
    reg += model.L1_in * tf.reduce_mean(tf.abs(model.W_in) * model.input_Connectivity)
    reg += model.L1_rec * tf.reduce_mean(tf.abs(model.W_rec) * model.rec_Connectivity)
    if model.dale_ratio:
        reg += model.L1_out * tf.reduce_mean(tf.matmul(tf.abs(model.W_out) * model.output_Connectivity, model.Dale_out))
    else:
        reg += model.L1_out * tf.reduce_mean(tf.abs(model.W_out) * model.output_Connectivity)

    # L2 weight regularization
    reg += model.L2_in * tf.reduce_mean(tf.square(tf.abs(model.W_in) * model.input_Connectivity))
    reg += model.L2_rec * tf.reduce_mean(tf.square(tf.abs(model.W_rec) * model.rec_Connectivity))
    if model.dale_ratio:
        reg += model.L2_out * tf.reduce_mean(tf.square(
            tf.matmul(tf.abs(model.W_out) * model.output_Connectivity, model.Dale_out)))
    else:
        reg += model.L2_out * tf.reduce_mean(tf.square(tf.abs(model.W_out) * model.output_Connectivity))

    # L2 firing rate regularization
    reg += model.L2_firing_rate * tf.reduce_mean(tf.square(tf.nn.relu(model.states)))

    # susillo regularization
    reg += model.sussillo_constant * model.sussillo_reg()

    return reg


def sussillo_reg(model):
    states = model.states

    reg = 0

    for state in states:
        dJr = tf.matmul(tf.nn.relu(state),
                        tf.matmul(tf.abs(model.W_rec) * model.rec_Connectivity, model.Dale_rec))
        reg += tf.reduce_sum(tf.square(dJr))

    return reg / (model.N_steps * model.N_batch)