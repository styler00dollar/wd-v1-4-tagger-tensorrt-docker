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
