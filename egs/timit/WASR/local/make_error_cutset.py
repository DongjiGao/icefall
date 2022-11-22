#!/usr/bin/env python3
# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

import argparse
import random
from pathlib import Path
from typing import List

from lhotse import load_manifest, CutSet
from lhotse.cut.base import Cut
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-cutset",
        type=str,
        help="supervision manifest that contains ground truth text",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        help="tokens",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help="error rate",
    )
    parser.add_argument(
        "--output-cutset",
        type=str,
        help="supervision manifest that contains modified error test",
    )
    parser.add_argument("--verbose-output", type=str, help="details of errors")
    return parser


def get_token_list(token_path: str) -> List:
    token_list = []
    with open(Path(token_path), "r") as tp:
        for line in tp.readlines():
            token = line.split()[0]
            assert token not in token_list
            token_list.append(token)
    return token_list


def make_error(
    cut: Cut,
    token_list: List,
    non_tokens: List,
    error_rate: float = 0.0,
) -> Cut:
    text = cut.supervisions[0].text
    text_list = text.split()
    new_text_list = []

    orig_verbose_list = []
    modified_verbose_list = []

    for token in text_list:
        new_token = token
        prob = random.random()
        if prob <= error_rate:
            while (
                new_token == token
                or new_token in non_tokens
                or new_token.startswith("#")
            ):
                new_token = random.choice(token_list)
            verbose_token = token.upper() 
            modified_verbose_token = new_token.upper()
        else:
            verbose_token = token
            modified_verbose_token = token

        orig_verbose_list.append(verbose_token)
        modified_verbose_list.append(modified_verbose_token)
        new_text_list.append(new_token)

    new_text = " ".join(new_text_list)
    cut.supervisions[0].text = new_text

    return cut, orig_verbose_list, modified_verbose_list


def main():
    non_tokens = set(("sil", "<UNK>", "<eps>", "<star>"))
    parser = get_parser()
    args = parser.parse_args()
    verbose_output = Path(args.verbose_output)

    cutset = load_manifest(Path(args.input_cutset))
    token_list = get_token_list(args.tokens)
    cuts = []

    with open(verbose_output, "w") as vo:
        for cut in cutset:
            modified_cut, orig_text_list, modified_text_list = make_error(
                cut=cut,
                token_list=token_list,
                non_tokens=non_tokens,
                error_rate=args.error_rate,
            )
            cuts.append(modified_cut)

            utt_id = cut.id
            orig_text = f"{utt_id} {' '.join(orig_text_list)}"
            modified_text = f"{utt_id} {' '.join(modified_text_list)}"
            vo.write(f"{orig_text}\n")
            vo.write(f"{modified_text}\n\n")

        output_cutset = CutSet.from_cuts(cuts)
        output_cutset.to_file(args.output_cutset)

if __name__ == "__main__":
    main()
