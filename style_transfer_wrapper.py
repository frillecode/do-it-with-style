import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import pickle
import ndjson
from wasabi import msg
from tqdm import tqdm


def preprocess_image(image_path, img_nrows, img_ncols):
    ''' Util function to open, resize and format pictures into appropriate tensors
    '''
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x, img_nrows, img_ncols):
    ''' Util function to convert a tensor into a valid image
    '''
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    ''' The gram matrix of an image tensor (feature-wise outer product)
    '''
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    ''' The "style loss" is designed to maintain the style of the reference image in the generated image. It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image
    '''
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    ''' An auxiliary loss function designed to maintain the "content" of the base image in the generated image
    '''
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x):
    ''' The 3rd loss function, total variation loss, designed to keep the generated image locally coherent
    '''
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def compute_loss(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, style_layer_names):
    ''' COmpute style transfer loss
    '''
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)


    # Weights of the different loss components
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, style_layer_names):
    ''' Decorator for loss and gradient computation (i.e. compile to make it fast)
    '''
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, style_layer_names)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

def main(style_reference_image_filename, style_reference_image_filepath, base_image_filename="white_noise", base_image_filepath="https://live.staticflickr.com/8465/8376267144_b0c41f8d65_b.jpg", iterations=4000):

    # Load images
    base_image_path = keras.utils.get_file(f"{base_image_filename}.jpg", base_image_filepath)
    style_reference_image_path = keras.utils.get_file(f"{style_reference_image_filename}.jpg", style_reference_image_filepath)
    result_prefix = style_reference_image_filename

    msg.info(f'Starting {result_prefix}')

    # Dimensions of the generated picture.
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    # List of layers to use for the style loss.
    style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    # The layer to use for the content loss.
    content_layer_name = "block5_conv2"


    # Training loop
    optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )
    base_image = preprocess_image(base_image_path, img_nrows, img_ncols)
    style_reference_image = preprocess_image(style_reference_image_path, img_nrows, img_ncols)
    combination_image = tf.Variable(preprocess_image(base_image_path, img_nrows, img_ncols))

    # set up logging
    losses = []
    for i in tqdm(range(1, iterations + 1)):
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image, feature_extractor, content_layer_name, style_layer_names
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            losses.append({
                'file': result_prefix,
                'iteration': i,
                'loss': loss
            })

    img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)
    fname = result_prefix + "_at_iteration_%d.png" % i
    keras.preprocessing.image.save_img(fname, img)

    with open(f'res/{result_prefix}', 'wb') as fp:
        pickle.dump(combination_image.numpy(), fp)
    
    with open(f'log/{result_prefix}', 'w') as fp:
        ndjson.dump(losses, fp)
    
    msg.good(f'Finished {result_prefix}')


if __name__ == '__main__':                                                                                                                         
    main(style_reference_image_filename ="rudolf", style_reference_image_filepath="https://www.wga.hu/art/a/aachen/rudolf2.jpg", iterations=200)

