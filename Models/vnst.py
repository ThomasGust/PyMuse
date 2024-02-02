#USES TENSORFLOW HUB MODEL NOT CUSTOM MODEL

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import PIL.Image
import os
import cv2
import shutil
import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)

def load_image(infilename, max_dim):
    img = tf.io.read_file(infilename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class VideoStylizer():


    def __init__(self):
        self.last_data_path = None
        self.last_stylized_data_path = None

    def split_video(self, video_path, output_dir):
        self.last_data_path = output_dir

        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f'{output_dir}/frame%d.jpg' % count, image)
            success, image = vidcap.read()
            print('Read a new frame : ', success)
            count += 1

    def stylize_video(self, video_data_dir_path, max_dim, output_dir, style_image_path):

        self.last_stylized_data_path = output_dir

        list_of_files_in_dir = os.listdir(video_data_dir_path)
        num_files_in_video_dir = len(list_of_files_in_dir)

        ordered_list_of_frame_numbers = []
        ordered_list_of_frame_full_names = []

        current_frame_number = 0
        for i in range(num_files_in_video_dir):
            ordered_list_of_frame_numbers.append(current_frame_number)
            current_frame_number += 1

        print(ordered_list_of_frame_numbers)

        current_frame_name_num = 0
        for i in range(num_files_in_video_dir):
            full_frame_name = f'frame{ordered_list_of_frame_numbers[current_frame_name_num]}'
            ordered_list_of_frame_full_names.append(full_frame_name)
            current_frame_name_num += 1

        print(ordered_list_of_frame_full_names)

        for i in ordered_list_of_frame_full_names:
            content_image = load_image(f'{video_data_dir_path}/{i}.jpg', max_dim=max_dim)
            style_image = load_image(style_image_path, max_dim=max_dim)

            print("About to load hub module")

            hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
            stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
            out_image = tensor_to_image(stylized_image)
            out_image.save(f'{output_dir}/{i}.jpg')
            print(f'successfully saved image : {i}')

    def stitch_video(self, video_data_dir_path, video_output_name, fps, video_size_x, video_size_y, format):


        if format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            format_extension = '.mp4'

        elif format == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            format_extension = '.avi'

        else:
            print('entered format was invalid. Reverting to mp4')
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            format_extension = '.mp4'

        out = cv2.VideoWriter(f'{video_output_name}{format_extension}', fourcc, fps, (video_size_x, video_size_y))

        files_in_data_dir = os.listdir(video_data_dir_path)
        length_of_video = len(files_in_data_dir)

        for i in range(length_of_video):
            img_path = f'{video_data_dir_path}/frame{i}.jpg'
            frame = cv2.imread(img_path)
            out.write(frame)
            print(f'Wrote frame : frame{i}')

        out.release()


    def fully_stylize_video(self, video_path, fps, style_image_path, max_dim,
                            format, video_size_x, video_size_y, new_video_name, remove_dirs=True):

        data_dir_name = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=30))
        stylized_data_dir_name = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=30))

        self.split_video(video_path=video_path, output_dir=data_dir_name)

        self.stylize_video(max_dim=max_dim, output_dir=stylized_data_dir_name, video_data_dir_path=data_dir_name,
                           style_image_path=style_image_path)

        self.stitch_video(format=format, fps=fps, video_size_x=video_size_x, video_size_y=video_size_y,
                          video_data_dir_path=stylized_data_dir_name, video_output_name=new_video_name)

        if remove_dirs:
            shutil.rmtree(path=data_dir_name)
            shutil.rmtree(path=stylized_data_dir_name)

