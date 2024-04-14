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


FROM python:3.11

ARG CLOUD_SDK_VERSION=google-cloud-cli-453.0.0-linux-x86_64.tar.gz

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${CLOUD_SDK_VERSION} && tar -xf ${CLOUD_SDK_VERSION} && ./google-cloud-sdk/install.sh

COPY requirements.txt .
RUN pip install -r requirements.txt


WORKDIR scripts 

COPY fasta_to_csv.py .
COPY tokenize_uniref_csv.py .
COPY process_uniref.sh .

ENV PATH="${PATH}:/google-cloud-sdk/bin"
ENTRYPOINT ["/bin/bash", "process_uniref.sh"]