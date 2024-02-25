from tqdm import tqdm
import argparse
import concurrent.futures
import cv2
import numpy as np
import onnxruntime
import os
import pandas as pd
import shutil

onnxruntime.set_default_logger_severity(3)
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_TENSORRT_CACHE_PATH"] = "."

# Define command line arguments
parser = argparse.ArgumentParser(description="Image inference using ONNX model")
parser.add_argument("--onnx", type=str, help="Path to ONNX model file")
parser.add_argument("--input", type=str, help="Path to folder containing input images")
parser.add_argument("--tag_mapping", type=str, help="Path to tag mapping csv file")
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Threshold for displaying tags"
)
parser.add_argument(
    "--corrupt", type=str, default="corrupt", help="Path to corrupt directory"
)
parser.add_argument("--num_threads", type=int, default=1, help="Amount of Threads")
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for image processing (currently only 1 supported)",
)
args = parser.parse_args()


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


def process_image(image_path, session, tag_df, threshold):
    try:
        img = cv2.imread(image_path)

        if img is None:
            os.rename(image_path, os.path.join(args.corrupt, image_file))
            return

        img = make_square(img, target_size=448)
        img = smart_resize(img, size=448)
        img = np.expand_dims(img, axis=0)

        if "joytag" in args.onnx:
            img = img.transpose((0, 3, 1, 2)) / 255

        if "wd" in args.onnx:
            probabilities = session.run(None, {"input_1:0": img})[0][0]
        elif "joytag" in args.onnx:
            probabilities = session.run(None, {"input": img})[0][0]
        else:
            raise Exception("Invalid model")

        tag_df["probs"] = probabilities

        found_tags = tag_df[tag_df["probs"] > threshold][["name"]]
        tag_names_str = ", ".join(found_tags["name"].tolist())

        txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        txt_path = os.path.join(os.path.dirname(image_path), txt_filename)

        with open(txt_path, "w") as f:
            f.write(tag_names_str)

    except Exception as e:
        print(e)
        shutil.move(image_path, os.path.join(args.corrupt, image_file))
        return


def process_images(images, session, tag_df, threshold):
    for image_file in images:
        if (
            image_file.endswith(".jpg")
            or image_file.endswith(".jpeg")
            or image_file.endswith(".png")
        ):
            image_path = os.path.join(args.input, image_file)
            process_image(image_path, session, tag_df, threshold)


sess_options = onnxruntime.SessionOptions()
providers = onnxruntime.get_available_providers()
print(providers)
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)

options = {}
options["device_id"] = 0
options["trt_engine_cache_enable"] = True
options[
    "trt_timing_cache_enable"
] = True  # Using TensorRT timing cache to accelerate engine build time on a device with the same compute capability
options["trt_engine_cache_path"] = "/workspace/"
options["trt_fp16_enable"] = True
options["trt_max_workspace_size"] = 7000000000  # ~7gb
options["trt_builder_optimization_level"] = 5

sessions = []
for i in range(args.num_threads):
    session = onnxruntime.InferenceSession(
        args.onnx,
        sess_options,
        providers=[
            ("TensorrtExecutionProvider", options),
        ],
    )
    sessions.append(session)

tag_df = pd.read_csv(args.tag_mapping, dtype={"tag_id": int})

futures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    pbar = tqdm(total=len(os.listdir(args.input)), position=0, leave=True)
    for i in range(0, len(os.listdir(args.input)), args.batch_size):
        image_batch = os.listdir(args.input)[i : i + args.batch_size]
        future = executor.submit(
            process_images,
            image_batch,
            sessions[i % args.num_threads],
            tag_df,
            args.threshold,
        )
        future.add_done_callback(lambda p: pbar.update(1))
        futures.append(future)
    # Use a dummy loop to ensure the bar stays active
    for _ in tqdm(
        concurrent.futures.as_completed(futures), total=len(futures), leave=False
    ):
        pass
    pbar.close()
