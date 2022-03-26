import os
from abc import ABC
from multiprocessing import Process
import tensorflow as tf
import time
import numpy as np
from img_utils import *

def vgg19_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def vgg16_layers(layer_names):
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def inceptionv3_layers(layer_names):
    vgg = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModelVGG19(tf.keras.models.Model, ABC):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModelVGG19, self).__init__()
        self.vgg19 = vgg19_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg19.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg19(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleContentModelVGG16(tf.keras.models.Model, ABC):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModelVGG16, self).__init__()
        self.vgg16 = vgg16_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg16.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
        outputs = self.vgg16(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleContentModelInceptionV3(tf.keras.models.Model, ABC):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModelInceptionV3, self).__init__()
        self.iv3 = inceptionv3_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.iv3.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.inception_v3.preprocess_input(inputs)
        outputs = self.iv3(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def VGG19StyleTransfer(content_image_tensor, style_image_tensor, content_layers=None,
                       style_layers=None
                       , return_im=True,
                       epochs=10,
                       steps_per_epoch=100,
                       style_weight=1e-2,
                       content_weight=1e4,
                       total_variation_weight=30,
                       tvl=True,
                       cl_addition=0,
                       style_loss_addition=0):
    if content_layers is None:
        content_layers = ['block5_conv2']
    if style_layers is None:
        style_layers = ['block1_conv1',
                        'block1_conv2',
                        'block2_conv1',
                        'block2_conv2',
                        'block3_conv1',
                        'block3_conv2',
                        'block4_conv1',
                        'block4_conv2',
                        'block5_conv1']
    style_image = style_image_tensor
    content_image = content_image_tensor

    # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg19_layers(style_layers)
    # style_outputs = style_extractor(style_image * 255)
    extractor = StyleContentModelVGG19(style_layers, content_layers)

    # results = extractor(tf.constant(content_image))

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers

        style_loss += style_loss_addition
        content_loss += cl_addition

        loss = style_loss + content_loss
        return loss

    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    """
    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
    """

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

            if tvl:
                loss += total_variation_weight * tf.image.total_variation(image)
            else:
                pass

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    del extractor
    del style_image
    del content_image
    del style_targets
    del content_targets
    del opt

    if return_im:
        return tensor_to_image(image)
    else:
        return image


def VGG16StyleTransfer(content_image_tensor, style_image_tensor, content_layers=None,
                       style_layers=None
                       , return_im=True,
                       epochs=10,
                       steps_per_epoch=100,
                       style_weight=1e-2,
                       content_weight=1e4,
                       total_variation_weight=30):
    if content_layers is None:
        content_layers = ['block5_conv2']
    if style_layers is None:
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']
    style_image = style_image_tensor
    content_image = content_image_tensor

    # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg16_layers(style_layers)
    # style_outputs = style_extractor(style_image * 255)
    extractor = StyleContentModelVGG16(style_layers, content_layers)

    # results = extractor(tf.constant(content_image))

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    """
    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
    """

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    if return_im:
        return tensor_to_image(image)
    else:
        return image