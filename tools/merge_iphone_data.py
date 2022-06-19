import glob
import argparse

parser = argparse.ArgumentParser(description="Data merger arg parser")
parser.add_argument("--dir", type=str, required=True, help="Root data directory")
args = parser.parse_args()

dirs = glob.glob(f"{args.dir}/exp-*")
for d in dirs:
    for side in ["left", "right"]:
        frames = glob.glob(f"{d}/{side}/zipped_frames/frame_*")
        print(f"{len(frames)} frames in {d}/{side}")
        for f in frames:
            # Create point cloud
            # Save to .bin
