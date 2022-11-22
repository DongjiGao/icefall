#!/bin/bash

# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

set -euo pipefail

stage=0
stop_stage=1000

log_dir=exp/log
cutset_dir=data/ssl
mkdir -p "${log_dir}"
mkdir -p "${cutset_dir}"

error_rates="0.1 0.3 0.5 0.7 0.9"

. ./cmd.sh
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Prepare data"
    ./prepare.sh --stage 2 --stop-stage 2 \
        --log-dir "${log_dir}"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Make errors to transcripts"
  for error_rate in ${error_rates}; do
    log "Error rate is: ${error_rate}"
    ./local/make_error_cutset.py \
      --input-cutset "${cutset_dir}/timit_cuts_TRAIN.jsonl.gz" \
      --tokens "data/lang_phone/tokens.txt" \
      --error-rate "${error_rate}" \
      --output-cutset "${cutset_dir}/timit_cuts_TRAIN_${error_rate}.jsonl.gz" \
      --verbose-output "${cutset_dir}/timit_${error_rate}_verbose.txt"
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Train WASR"
    ${cuda_cmd} "${log_dir}/train.log" \
        python tdnn_lstm_ctc/train.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Decode"
    ${cuda_cmd} "${log_dir}/decode.log" \
        python tdnn_lstm_ctc/decode.py
fi

exit 0
