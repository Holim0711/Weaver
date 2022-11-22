FROM pytorch/pytorch:latest
RUN pip install pytorch-lightning adabelief-pytorch
COPY . /opt/Weaver-pytorch
WORKDIR /opt/Weaver-pytorch
RUN pip install .
WORKDIR /workspace