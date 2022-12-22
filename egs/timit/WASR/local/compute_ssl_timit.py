#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                  Mingshuang Luo)
#              2022  Johns Hopkins University (author: Dongji Gao)   
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file computes SSL features of the TIMIT dataset.
It looks for manifests in the directory data/manifests.

The generated SSL features are saved in data/ssl.
"""

import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, S3PRLSSL, S3PRLSSLConfig, NumpyFilesWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_ssl_timit():
    src_dir = Path("data/manifests")
    output_dir = Path("data/ssl")
    num_jobs = 1

    dataset_parts = (
        "TRAIN",
        "DEV",
        "TEST",
    )
    prefix = "timit"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = S3PRLSSL(
        S3PRLSSLConfig(ssl_model="wav2vec2", device="cuda")
    )

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_file = output_dir / f"{prefix}_cuts_{partition}.{suffix}"
            if cuts_file.is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if partition == "TRAIN":
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cut_set = cut_set.trim_to_supervisions()
            cut_set = cut_set.to_eager()
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=NumpyFilesWriter,
            )
            cut_set.to_file(cuts_file)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_ssl_timit()
