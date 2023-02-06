from keras.layers import Dense, Input, Lambda
import keras


def distillation_loss(y_true, y_pred, temperature, alpha):
    # temperature = 0.8
    # alpha = 0.8
    y_pred_soft = Lambda(lambda x: x / temperature)(y_pred)
    # print('y_pred_soft',y_pred_soft)
    # print(y_true)
    # print(y_pred)
    # print(y_pred_soft)

    return alpha * keras.losses.categorical_crossentropy(y_true, y_pred) + (1-alpha) * keras.losses.categorical_crossentropy(y_true, y_pred_soft)
