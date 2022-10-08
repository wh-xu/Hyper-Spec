FROM nvidia/cuda:11.2.0-runtime-ubuntu18.04

# Update and install mini-conda
RUN apt-get update
RUN apt-get install -y time wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libboost-all-dev git mercurial subversion && \
        apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./docker/requirements.yaml /tmp/requirements.yaml
RUN conda update conda \
    && conda env create --name hyper-spec -f /tmp/requirements.yaml

RUN echo "conda activate hyper-spec" >> ~/.bashrc
ENV PATH /opt/conda/envs/hyper-spec/bin:$PATH
ENV CONDA_DEFAULT_ENV $hyper-spec

# Clone the code
RUN git clone https://github.com/wh-xu/Hyper-Spec.git 

# set working dir
WORKDIR /Hyper-Spec

# Install and build  
RUN bash ./install.sh 
