import PIL.Image
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import numpy as np
import tensorflow as tf


def modifyImageBscc(imageData, brightness, sharpness, contrast, color):
    """Update with brightness, sharpness, contrast and color."""
    imageData = Image.fromarray(imageData)
    brightnessMod = ImageEnhance.Brightness(imageData)
    imageData = brightnessMod.enhance(brightness)

    sharpnessMod = ImageEnhance.Sharpness(imageData)
    imageData = sharpnessMod.enhance(sharpness)

    contrastMod = ImageEnhance.Contrast(imageData)
    imageData = contrastMod.enhance(contrast)

    colorMod = ImageEnhance.Color(imageData)
    imageData = colorMod.enhance(color)

    return np.array(imageData)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img