"""
Method adapted from breast_cancer_classifier function `run_producer` by
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
import random

import argparse

from src.heatmaps.run_producer import produce_heatmaps
from src.heatmaps.run_producer import load_model

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmaps')
    parser.add_argument('--exam-list-path', default='sample_output/exam_list.pkl')
    parser.add_argument('--image-path', default='sample_output/cropped_images')
    parser.add_argument('--output-heatmap-path', default='sample_output/heatmaps')
    parser.add_argument('--model-path', default='models/sample_patch_model.p')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--use-hdf5', choices=[False, True], default=False)
    parser.add_argument('--device-type', choices=['gpu', 'cpu'], default='cpu')
    parser.add_argument('--gpu-number', type=int, default=0)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    # Set the seed
    random.seed(args.seed)

    params = dict(
        device_type=args.device_type,
        gpu_number=args.gpu_number,
        patch_size=256,
        stride_fixed=70,
        more_patches=5,
        minibatch_size=args.batch_size,
        seed=args.seed,
        initial_parameters=args.model_path,
        input_channels=3,
        number_of_classes=4,
        data_file=args.exam_list_path,
        original_image_path=args.image_path,
        save_heatmap_path=[os.path.join(args.output_heatmap_path, 'heatmap_malignant'),
                           os.path.join(args.output_heatmap_path, 'heatmap_benign')],
        heatmap_type=[0, 1],
        use_hdf5=args.use_hdf5
    )

    # Get model
    model, device = load_model(params)

    # Generate heatmaps in the chosen format
    produce_heatmaps(model, device, params)
