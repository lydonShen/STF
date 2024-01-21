# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os
from fcos_core.config import cfg
from predictor import COCODemo

import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="/root/codecar1/configs/fcos/fcos_imprv_R_50_FPN_1x_plane.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="/root/codecar1/demo/images/model_final.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="/root/codecar1/demo/images/val2014/",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [
            0.7364850640296936, 0.7193474769592285, 0.6652705669403076, 0.6768263578414917, 
            0.7614352107048035, 0.7242012023925781, 0.7344110608100891, 0.6862906813621521,0.7614352107048035, 0.7242012023925781, 0.7344110608100891, 0.6862906813621521]

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imwrite(im_name, composite)
    print("Press any keys to exit ...")
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

