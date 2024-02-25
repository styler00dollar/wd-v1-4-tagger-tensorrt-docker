# wd-v1-4-tagger-tensorrt-docker
```
# build with my docker
docker build -t styler00dollar/onnxruntime-trt:latest -f Dockerfile.tensorrt .
```
```
# download docker from dockerhub
docker pull styler00dollar/onnxruntime-trt:latest
```
```
# run docker
docker run --gpus all -it -v /home/user/Downloads/:/workspace/ styler00dollar/onnxruntime-trt:latest 

# example usage
python tag.py --onnx wd-v1-4-swinv2-tagger-v2_448_0.6854.onnx --input test_data/ --tag_mapping selected_tags.csv --threshold 0.90 --num_threads=16 --batch_size=1
```
Benchmarks:

Tests were done 10000 images with 448px png images. Average speed of running this script on a 4090. Further speed optimization could be done by batching data.

Images can be generated with:
```python
import numpy as np
from PIL import Image

for i in range(1000):
    random_image = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
    img = Image.fromarray(random_image)
    img.save(f"random_image_{i}.png")
```

TensorRT settings:
```
options["trt_engine_cache_enable"] = True
options["trt_timing_cache_enable"] = True 
options["trt_fp16_enable"] = True
options["trt_max_workspace_size"] = 7000000000  # ~7gb
options["trt_builder_optimization_level"] = 5
```

| Model                                            | TensorrtExecutionProvider (fp16) | TensorrtExecutionProvider+16 threads (fp16) |
| ------------------------------------------------ |:-------------------------------: |:-------------------------------------------:|
| wd-v1-4-swinv2-tagger-v2_448_0.6854_sim.onnx     | 181.81it/s (55 seconds)          | 277.77it/s (36 seconds)                     | 
| wd-v1-4-moat-tagger-v2_448_0.6911_sim.onnx       | 192.30it/s (52 seconds)          | 277.77it/s (36 seconds)                     |
| wd-v1-4-convnextv2-tagger-v2_448_0.6862_sim.onnx | 108.69it/s (1:32 min)            | 277.77it/s (36 seconds)                     |
| wd-v1-4-vit-tagger-v2_448_0.6770_sim.onnx        | 169.49it/s (59 seconds)          | 277.77it/s (36 seconds)                     |
| joytag_clamp_normalized_448px_sim.onnx           | 212.76it/s (47 seconds)          | freezing
