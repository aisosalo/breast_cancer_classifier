"""
Method adapted from breast_cancer_classifier function `crop_mammogram` by
Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin,
Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh,
Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao,
Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema,
Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy,
Kyunghyun Cho, and Krzysztof J. Geras , which is licensed under a GNU Affero General Public License v3.0.
See: https://raw.githubusercontent.com/nyukat/breast_cancer_classifier/master/LICENSE
"""

import sys
import os

from multiprocessing import Pool

import argparse

from functools import partial

import src.utilities.pickling as pickling
import src.utilities.data_handling as data_handling

from src.cropping.crop_mammogram import crop_mammogram_one_image_short_path

print(sys.version, sys.platform, sys.executable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--input_data_folder', default='sample_data/images')
    parser.add_argument('--output_data_folder', default='sample_output/cropped_images')
    parser.add_argument('--exam_list_path', default='sample_data/exam_list_before_cropping.pkl')
    parser.add_argument('--cropped_exam_list_path', default='sample_output/cropped_exam_list.pkl')
    parser.add_argument('--num_processes', default=None)
    parser.add_argument('--num_iterations', default=100, type=int)
    parser.add_argument('--buffer_size', default=50, type=int)
    args = parser.parse_args()

    exam_list = pickling.unpickle_from_file(args.exam_list_path)

    image_list = data_handling.unpack_exam_into_images(exam_list)

    os.makedirs(args.output_data_folder, exist_ok=True)

    crop_mammogram_one_image_func = partial(
        crop_mammogram_one_image_short_path,
        input_data_folder=args.input_data_folder,
        output_data_folder=args.output_data_folder,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )

    with Pool(args.num_processes) as pool:
        cropped_image_info = pool.map(crop_mammogram_one_image_func, image_list)

    window_location_dict = dict([x[0] for x in cropped_image_info])
    rightmost_points_dict = dict([x[1] for x in cropped_image_info])
    bottommost_points_dict = dict([x[2] for x in cropped_image_info])
    distance_from_starting_side_dict = dict([x[3] for x in cropped_image_info])

    data_handling.add_metadata(exam_list, "window_location", window_location_dict)
    data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
    data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
    data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)

    pickling.pickle_to_file(args.cropped_exam_list_path, exam_list)
