    layer_name = layer

    nb_classes = 1000
    target_layer = lambda x: x*K.one_hot([prediction_index], nb_classes)
    x = Lambda(target_layer, output_shape=lambda input_shape: input_shape)(estimator.output)
    model = Model(inputs=estimator.input, outputs=x)
    model.summary()

    loss = K.sum(model.output)
    # TODO: check if layer not found
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output

    # grads = tf.gradients(tensor, var_list)
    grads = K.gradients(loss, [conv_output])
    grads = [grad if grad is not None else K.zeros_like(var) 
    for var, grad in zip([conv_output], grads)][0]

    grads = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([x]) # ***
    # output, grads_val = output[0, :], grads_val[0, :, :, :]
    # weights = np.mean(grads_val, axis = (0, 1))
    # cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    # for i, w in enumerate(weights):
    #     cam += w * output[:, :, i]
    # cam = cv2.resize(cam, (224, 224))
    # cam = np.maximum(cam, 0)
    # heatmap = cam / np.max(cam)
    # image = image[0, :]
    # image -= np.min(image)
    # image = np.minimum(image, 255)
    # cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # cam = np.float32(cam) + np.float32(image)
    # cam = 255 * cam / np.max(cam)
    # return np.uint8(cam), heatmap
