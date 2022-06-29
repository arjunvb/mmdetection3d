import glob
import argparse
import json
import numpy as np
import os
import ray
import cv2
from scipy.spatial.transform import Rotation as R

ray.init()

# Get utils on path
import sys

sys.path.append("../../")
from utils import data_utils, calc_utils, consts

# random seed
import random

random.seed(4)

MRCNN_CAR = 3


@ray.remote
def save_frame(out_dir, idx, img_path, dep_path, mask_path):
    sys.path.append("../../")
    from utils import data_utils, calc_utils, consts, img_utils

    img = data_utils.create_rgb_frame(img_path, downsample=True)
    dep, _ = data_utils.get_buffer(dep_path, np.float32)
    pts, colors, pix2pts = calc_utils.create_point_cloud(
        img, dep, consts.EXAMPLE_INTRINSIC
    )

    # Save pcl
    with open(f"{out_dir}/pcl/{idx:09}.bin", "w") as x:
        pts.tofile(x)

    with open(f"{out_dir}/col/{idx:09}.bin", "w") as x:
        colors.tofile(x)

    # Create img symlink
    img_link = f"{out_dir}/img/{idx:09}.jpg"
    if not os.path.exists(img_link):
        os.symlink(img_path, img_link)

    # Create labels (based on mask-rcnn labels)
    f_lab = open(f"{out_dir}/lab/{idx:09}.txt", "w")
    with open(mask_path) as j:
        masks = json.load(j)
        for i, class_id in enumerate(masks["class_ids"]):
            if class_id == MRCNN_CAR:
                roi = masks["rois"][i]
                score = masks["scores"][i]
                pix = data_utils.downsample_indices(masks["locs"][i])

                blob = np.array([pix2pts[p] for p in zip(pix[0], pix[1])])
                with open(f"{out_dir}/pcl/blob{i}_{idx:09}.bin", "w") as x:
                    blob.tofile(x)
                cblob = [img[px, py, :] for px, py in zip(pix[0], pix[1])]
                cblob = np.array(cblob).astype(np.float32) / 255.0
                with open(f"{out_dir}/pcl/cblob{i}_{idx:09}.bin", "w") as x:
                    cblob.tofile(x)

                # Create bounding box
                center, extent, rot = calc_utils.create_3d_bbox(blob)
                r = R.from_matrix(rot.copy())
                ry = r.as_rotvec()[1]

                # Convert to str
                roi_str = " ".join(map(str, list(roi)))
                extent_str = " ".join(map(str, list(extent)))
                center_str = " ".join(map(str, list(center)))

                # Write label to file
                f_lab.write(
                    "Car 0 0 2.0 {} {} {} {} {}\n".format(
                        roi_str, extent_str, center_str, ry, score
                    )
                )

        if len(masks["class_ids"]) > 0:
            filepath = mask_path.split("masks/")[0]
            fidx = data_utils.get_frame_id(dep_path)
            img_overlay, _ = img_utils.overlay_mask(filepath, fidx, 0)
            cv2.imwrite(f"{out_dir}/img/overlay_{idx:09}.jpg", img_overlay)

    f_lab.close()

    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data merger arg parser")
    parser.add_argument("--dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    # Create out dir if needed
    dir_train_pcl = f"{args.out}/training/pcl/"
    if not os.path.exists(dir_train_pcl):
        os.makedirs(dir_train_pcl)

    dir_train_col = f"{args.out}/training/col/"
    if not os.path.exists(dir_train_col):
        os.makedirs(dir_train_col)

    dir_train_img = f"{args.out}/training/img/"
    if not os.path.exists(dir_train_img):
        os.makedirs(dir_train_img)

    dir_train_lab = f"{args.out}/training/lab/"
    if not os.path.exists(dir_train_lab):
        os.makedirs(dir_train_lab)

    dir_test_pcl = f"{args.out}/testing/pcl/"
    if not os.path.exists(dir_test_pcl):
        os.makedirs(dir_test_pcl)

    dir_test_col = f"{args.out}/testing/col/"
    if not os.path.exists(dir_test_col):
        os.makedirs(dir_test_col)

    dir_test_img = f"{args.out}/testing/img/"
    if not os.path.exists(dir_test_img):
        os.makedirs(dir_test_img)

    dir_test_lab = f"{args.out}/testing/lab/"
    if not os.path.exists(dir_test_lab):
        os.makedirs(dir_test_lab)

    dir_is = f"{args.out}/ImageSets/"
    if not os.path.exists(dir_is):
        os.makedirs(dir_is)

    # Global idx
    idx = 0

    # ImageSet files
    f_train = open(f"{args.out}/ImageSets/train.txt", "w")
    f_val = open(f"{args.out}/ImageSets/val.txt", "w")
    f_test = open(f"{args.out}/ImageSets/test.txt", "w")

    dirs = glob.glob(f"{args.dir}/exp-*")
    for d in dirs[:1]:
        for side in ["left", "right"]:
            frames = glob.glob(f"{d}/{side}/zipped_frames/frame_*.dep")
            print(f"{len(frames)} frames in {d}/{side}")

            ret = []
            for i, f in enumerate(frames):
                frame_id = data_utils.get_frame_id(f)

                # Create point cloud and save to .bin
                img_path = f"{d}/{side}/zipped_frames/frame_{frame_id}.jpg"
                mask_path = f"{d}/{side}/masks/{frame_id}.json"
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    fname = f"{idx+i:09}"
                    dataset = random.choices(
                        ["training", "val", "testing"], weights=(70, 10, 20)
                    )
                    if dataset[0] == "training":
                        folder = "training"
                        f_train.write(f"{fname}\n")
                    elif dataset[0] == "val":
                        folder = "training"
                        f_val.write(f"{fname}\n")
                    else:
                        folder = "testing"
                        f_test.write(f"{fname}\n")
                    ret += [
                        save_frame.remote(
                            f"{args.out}/{folder}/", idx + i, img_path, f, mask_path
                        )
                    ]
                else:
                    print(f"idx {idx + i} does not exist")

            ret = ray.get(ret)
            idx += len(frames)

    f_train.close()
    f_val.close()
    f_test.close()
