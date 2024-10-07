# Use an official CUDA runtime as the base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.5"

# Add Conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Initialize Conda
RUN $CONDA_DIR/bin/conda init bash

# Clone the GaussianPro repository
RUN git clone https://github.com/kcheng1021/GaussianPro.git --recursive

# Set the working directory
WORKDIR /GaussianPro

# Copy environment.yml into the Docker image
COPY environment.yml .

# Create the Conda environment and install additional packages
# RUN /opt/conda/bin/conda env create -f environment.yml

# Activate the environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate gaussianpro" >> /root/.bashrc

# Create the Conda environment, install packages, and clean up in one RUN command
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    conda activate gaussianpro && \
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge && \
    pip install --upgrade pip && \
    pip install ./submodules/Propagation && \
    pip install ./submodules/diff-gaussian-rasterization && \
    pip install ./submodules/simple-knn && \
    conda clean -afy"

# Copy the entrypoint script into the Docker image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint to the script that activates the Conda environment
ENTRYPOINT ["/entrypoint.sh"]

# Set the default command to bash to keep the container running
CMD ["/bin/bash"]
