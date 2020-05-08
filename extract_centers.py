"""
Method adapted from breast_cancer_classifier function `get_optimal_centers` by
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

import argparse

import src.utilities.pickling as pickling
import src.utilities.data_handling as data_handling

from src.optimal_centers.get_optimal_centers import get_optimal_centers

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and extract optimal centers')
    parser.add_argument('--cropped-exam-list-path', default='sample_output/cropped_exam_list.pkl')
    parser.add_argument('--data-prefix', default='sample_output/cropped_images')
    parser.add_argument('--output-exam-list-path', default='sample_output/exam_list.pkl')
    parser.add_argument('--num-processes', default=None)
    args = parser.parse_args()

    exam_list = pickling.unpickle_from_file(args.cropped_exam_list_path)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)

    optimal_centers = get_optimal_centers(  # uses Pool
        data_list=data_list,
        data_prefix=args.data_prefix,
        num_processes=args.num_processes
    )

    data_handling.add_metadata(exam_list, "best_center", optimal_centers)

    output_exam_list_path = args.output_exam_list_path

    os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
    pickling.pickle_to_file(output_exam_list_path, exam_list)
