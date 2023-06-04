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
python tag.py --onnx wd-v1-4-swinv2-tagger-v2.onnx --input test_data/ --tag_mapping selected_tags.csv --threshold 0.90 [--corrupt corrupt]
```
Benchmarks:

Tests were done on a large number of random images. Benchmarks currently only for fp32 since I don't know if fp16 is viable with these models. Average speed of running this script on a 4090.

| Model                                         | TensorrtExecutionProvider (fp32)  |
| --------------------------------------------- |:---------------------------------:|
| wd-v1-4-swinv2-tagger-v2_448_0.6854.onnx      | 11.71img/s                        |
| wd-v1-4-moat-tagger-v2_448_0.6911.onnx        | 11.93img/s                        |
| wd-v1-4-convnext-tagger-v2_448_0.6810.onnx    | 11.8img/s                         |
| wd-v1-4-convnextv2-tagger-v2_448_0.6862.onnx  | 11.65img/s                        |
| wd-v1-4-vit-tagger-v2_448_0.6770.onnx         | 12.12img/s                        |
