# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_tpuvm_20240414
ARG CLOUD_SDK_VERSION=google-cloud-cli-453.0.0-linux-x86_64.tar.gz

RUN apt-get update -y
RUN apt-get install numactl 
RUN apt-get install vim-tiny -y

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${CLOUD_SDK_VERSION} && tar -xf ${CLOUD_SDK_VERSION} && ./google-cloud-sdk/install.sh

RUN git clone https://github.com/huggingface/transformers.git
WORKDIR transformers
RUN pip3 install -e . &&  pip install datasets accelerate evaluate scikit-learn

COPY modeling_esm.py src/transformers/models/esm
COPY run_esm2_spmd.py .

ENV PATH="${PATH}:/google-cloud-sdk/bin"