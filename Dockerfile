# Use the official Conda base image with Python 3.7
# full image including full CUDA toolkit with nvcc compiler and headers
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel 

# Set working directory
WORKDIR /SDCNet

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Ensure CUDA is properly visible
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Conda installs
RUN conda install -c conda-forge timm==1.0.15 -y

# Install PyTorch with CUDA from pip explicitly
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN ls -l /usr/local/cuda && ls -l /usr/local/cuda/lib64 && ls -l /usr/local/cuda/include && nvcc --version

# Other python installs
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"

COPY . .

ENTRYPOINT ["bash", "-c"]
# ENTRYPOINT ["python"]

CMD ["python", "test_evt_tlc.py", "--imagenet_model", "Flash_InternImage", "--cfg", "flash_intern_image_b_1k_224.yaml", "--rcan_model", "'SAFMN'", "--base_size", "3350", "kernel_size", "5", "--model_save_dir", "./output_result", "-tlc_on", "on", "--input_ensemble", "True", "--ckpt_path", "output/backbone/Flash_InternImage/22.57_f800_512_1e4_ema_mixper_woGAN/epoch800.pkl", "--hazy_data", "NHNH2", "--cropping", "4"]

# NOTE: DCNv4 needs to be pip installed on run


# TO BUILD IMAGE
# hare build -t ceh94/sdc .

# # docker system prune ## run every so often to clear out system

# # TO RUN IMAGE FOR TESTING
# hare run --rm --gpus '"device=0,1,2"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/weights,target=/MyModel/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/output_result,target=/MyModel/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/input_data,target=/MyModel/input_data \
# ceh94/mymodel \
# test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_result --hazy_data NHNH2 --cropping 1 --dataset_name NHNH2

# # # TO RUN IMAGE FOR TRAINING
# hare run --rm --gpus '"device=5,6"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/weights,target=/MyModel/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/output_result,target=/MyModel/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/data,target=/MyModel/data \
# --mount type=bind,source=/mnt/faster0/ceh94/MyModel/check_points,target=/MyModel/check_points \
# ceh94/mymodel \
# train.py --data_dir training_data --imagenet_model SwinTransformerV2 --cropping 1 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml -train_batch_size 8 --model_save_dir check_points -train_epoch 8005 --save_prefix NH2