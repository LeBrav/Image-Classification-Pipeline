import keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

def reverse_preprocess_input(x, original_type):
    x *= 0.5
    x += 0.5
    x *= 255
    x = x.astype(original_type)
    return x


def grad_cam(model, input_tensor, class_index, layer_name):
    layer_output = model.get_layer(layer_name).output

    loss = model.output[:, class_index]
    grads = K.gradients(loss, layer_output)[0]
    gradient_function = K.function([model.input], [layer_output, grads])

    output, grads_val = gradient_function([input_tensor])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = np.maximum(cam, 0)
    cam = np.uint8(255 * cam / np.max(cam))

    cam = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_LINEAR)
    cam = cam / np.max(cam)
    return cam


def func_inference_CAM(model, data, cfg):
    prediction = model.predict(data[0])
    classes = cfg["inference_model"]["classes"]

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    fig1, axs1 = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(16):
        row = i // 4
        col = i % 4
        input_img = np.expand_dims(data[0][i], axis=0)
        class_index = np.argmax(prediction[i])
        cam = grad_cam(model, input_img, class_index, cfg["inference_model"]["output_layer"])

        # data_img = reverse_preprocess_input(data[0][i], "uint8")

        title = "Pred: " + str(classes[class_index]) + " - GT: " + str(classes[np.argmax(data[1][i])])

        axs[row, col].imshow(data[0][i])
        axs[row, col].imshow(cam, cmap="jet", alpha=0.3)
        axs[row, col].axis("off")
        axs[row, col].set_title(title)

        axs1[row, col].imshow(data[0][i])
        axs1[row, col].axis("off")
        axs1[row, col].set_title(title)

    plt.tight_layout()
    plt.show()
