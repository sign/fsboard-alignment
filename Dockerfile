# Image for the full run: features, training, inference and the two output files.
#
#   docker build -t fsboard-alignment .
#   docker run --rm --gpus all --ipc host --env-file .env -v /data:/data -v $PWD:/work fsboard-alignment

# NVIDIA's PyTorch build: torch with CUDA, also for arm64 machines such as the DGX Spark
FROM nvcr.io/nvidia/pytorch:26.10-py3

# The package, installed in place so that a mounted checkout can replace it
WORKDIR /src
COPY pyproject.toml README.md ./
COPY fsboard_alignment fsboard_alignment
RUN pip install --no-cache-dir -e .

# Fail the build, not the run, if an install replaced the image's torch or broke video decoding
RUN python -c "import torch, av, simple_video_utils, fsboard_alignment; print(torch.__version__, av.__version__)"

# Run from the mounted working directory: weights/ is read there,
# and features/, runs/ and output/ are written there
WORKDIR /work

# `docker run ... fsboard-alignment` does the full run;
# `docker run ... fsboard-alignment fsboard_alignment.train all` runs one step
ENTRYPOINT ["python", "-m"]
CMD ["fsboard_alignment"]
