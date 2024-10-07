docker run --rm --gpus all -it \
    --entrypoint /bin/bash \
    -v your_dataset_path:/GaussianPro/datasets \
    gaussian-pro
