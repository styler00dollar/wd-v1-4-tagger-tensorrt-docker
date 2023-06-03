import onnxruntime
import argparse
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_TENSORRT_CACHE_PATH"] = "."

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model file")
parser.add_argument(
    "--input", type=str, required=True, help="Path to folder containing input images"
)
parser.add_argument(
    "--tag_mapping", type=str, required=True, help="Path to tag mapping file"
)
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Threshold for displaying tags"
)
parser.add_argument(
    "--corrupt", type=str, default="corrupt", help="Path to corrupt directory"
)

# Parse command line arguments
args = parser.parse_args()

# Load ONNX model
sess_options = onnxruntime.SessionOptions()
providers = onnxruntime.get_available_providers()
print(providers)
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)
session = onnxruntime.InferenceSession(
    args.onnx,
    sess_options,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

# Check if the --corrupt argument is specified
if args.corrupt == "corrupt":
    # Set a default location for the corrupt folder
    default_corrupt_location = os.path.join(os.getcwd(), "corrupt")
    args.corrupt = default_corrupt_location

# Create the corrupt folder if it doesn't exist
if not os.path.exists(args.corrupt):
    os.makedirs(args.corrupt)


# https://github.com/SmilingWolf/SW-CV-ModelZoo/blob/main/Utils/dbimutils.py


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = (255, 255, 255)
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)


# Load images from input folder and process them
for image_file in tqdm(os.listdir(args.input)):
    if (
        image_file.endswith(".jpg")
        or image_file.endswith(".jpeg")
        or image_file.endswith(".png")
    ):
        image_path = os.path.join(args.input, image_file)

        try:
            img = cv2.imread(image_path)
            if img is None:
                # Move the unreadable/corrupt image file to a different folder
                os.rename(image_path, os.path.join(args.corrupt, image_file))
                continue
        except Exception as e:
            # Move the unreadable/corrupt image file to a different folder
            os.rename(image_path, os.path.join(args.corrupt, image_file))
            continue

        # Preprocess image
        img = make_square(img, target_size=448)
        img = smart_resize(img, size=448)
        img = np.expand_dims(img, axis=0)

        # Apply model and extract tags
        tag_df = pd.read_csv(args.tag_mapping, dtype={"tag_id": int})
        probabilities = session.run(None, {"input_1:0": img})[0][0]
        tag_indices = np.argsort(probabilities)[::-1]
        tag_df["probs"] = probabilities
        found_tags = tag_df[tag_df["probs"] > args.threshold][["name"]]

        # Print results
        tag_names_str = ", ".join(found_tags["name"].tolist())

        # Save tag string as file in the same directory as the image file
        txt_filename = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(os.path.dirname(image_path), txt_filename)
        with open(txt_path, "w") as f:
            f.write(tag_names_str)
