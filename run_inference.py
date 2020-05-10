"""
Method adapted from breast_cancer_classifier function `run_model` by
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

import pandas as pd

import src.utilities.pickling as pickling

from src.constants import LABELS, MODELMODES
from src.modeling.run_model import load_model, run_model

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--model-mode', default=MODELMODES.VIEW_SPLIT, type=str)
    parser.add_argument('--model-root', default='models/')
    parser.add_argument('--model', choices=['sample_image_model.p', 'sample_imageheatmaps_model.p'], default='sample_imageheatmaps_model.p')
    parser.add_argument('--data-path', default='sample_output/exam_list.pkl')
    parser.add_argument('--image-path', default='sample_output/cropped_images/')
    parser.add_argument('--heatmaps-path', default='sample_output/heatmaps/')
    parser.add_argument('--output-path', default='sample_output/')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', choices=[False, True], default=True)
    parser.add_argument('--use-augmentation', choices=[False, True], default=True)
    parser.add_argument('--use-hdf5', choices=[True], default=True)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--device-type', choices=['gpu', 'cpu'], default='cpu')
    parser.add_argument('--gpu-number', default=0, type=int)
    parser.add_argument('--save-output', choices=[False, True], default=True)
    args = parser.parse_args()

    params = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmaps_path": args.heatmaps_path,
        "use_hdf5": args.use_hdf5,
        "model_mode": args.model_mode,
        "model_path": os.path.join(args.model_root, args.model),
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    exam_list = pickling.unpickle_from_file(args.data_path)

    model, device = load_model(params)

    predictions = run_model(model, device, exam_list, params)

    predictions = pd.DataFrame(predictions, columns=LABELS.LIST)
    if args.save_output:
        filename = "image_heatmaps_predictions.csv" if args.use_heatmaps else "image_only_predictions.csv"
        predictions.to_csv(os.path.join(args.output_path, filename), index=False, float_format='%.4f')
