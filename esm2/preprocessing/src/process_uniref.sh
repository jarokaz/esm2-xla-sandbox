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

#!/usr/bin/bash -x

ls -la /gcs/artifact-repository
exit 0

CSV_FOLDER=/tmp/staging/uniref/sharded
TOKENIZED_FOLDER=/tmp/staging/uniref/tokenized

gcloud storage ls ${GCS_OUTPUT}/processed/csv
if [[ $? -ne 0 ]]; then
  echo "Downloading and sharding Uniref50"
  python fasta_to_csv.py \
  --source https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz \
  --output_dir ${CSV_FOLDER} \
  --max_records_per_partition ${MAX_RECORDS_PER_PARTITION}

  if [[ $? -eq 0 ]]; then
    echo "Copying Uniref shards to ${GCS_OUTPUT}/processed"
    gcloud storage rsync --recursive --no-clobber ${CSV_FOLDER} ${GCS_OUTPUT}/processed
  else
    echo "Uniref processing failed"
  fi

else 
   echo "Sharded CSV files exist. Skipping the download step."
fi

if [[ ! -d /tmp/staging/uniref ]]; then
  echo "Copying sharded files from GCS"
  gcloud storage rsync --recursive --no-clobber  ${GCS_OUTPUT}/processed ${CSV_FOLDER} 
fi

echo "*******************"
ls -la ${CSV_FOLDER}

echo "Tokenizing shards"
python tokenize_uniref_csv.py \
--tokenizer_name facebook/esm2_t30_150M_UR50D \
--max_seq_length ${MAX_SEQ_LENGTH} \
--preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
--line_by_line ${LINE_BY_LINE} \
--train_size ${TRAIN_SIZE} \
--validation_size ${VALIDATION_SIZE} \
--test_size ${TEST_SIZE} \
--input_dir ${CSV_FOLDER}/csv \
--output_dir ${TOKENIZED_FOLDER} 

if [[ $? -eq 0 ]]; then
  echo "Copying tokenized shards to ${GCS_OUTPUT}/tokenized"
  gcloud storage rsync --recursive --no-clobber ${TOKENIZED_FOLDER} ${GCS_OUTPUT}/tokenized
else
  echo "Tokenization failed"
fi
