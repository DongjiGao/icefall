#!/usr/bin/env python3

# Copyright    2024 Johns Hopkins University    (author: Dongji Gao)

from pathlib import Path
from typing import List, Tuple, Union

import k2
import sentencepiece as spm
import torch

from icefall.utils import str2bool


class TransducerTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        blank_id=0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:
                - bpe.model
                - words.txt
          blank_id:
            blank token id.
          device:
            It indicates CPU or CUDA.
        """
        lang_dir = Path(lang_dir)
        bpe_model = lang_dir / "bpe.model"

        sp = spm.SentencePieceProcessor()
        self.sp = sp.load(str(bpe_model))

        self.device = device
        self.blank_id = blank_id

    def compile(
        self,
        texts: List[str],
    ) -> k2.Fsa:
        pass

    def convert_transcript_to_fsa(
        self,
        unit_ids: List[int],
        num_frames: int,
    ) -> k2.Fsa:
        U = len(unit_ids)
        T = num_frames

        num_states = (U + 1) * T
        num_forward_arcs = (U + 1) * (T - 1)
        num_unit_arcs = U * T

        forward_arcs_from_states, forward_arcs_to_states = self.get_forward_arcs_states(
            num_forward_arcs, U
        )
        unit_arcs_from_states, unit_arcs_to_states = self.get_unit_arcs_states(
            num_states, U, T
        )
        # assemble arcs
        arcs = self.assemble_arcs(
            unit_ids,
            self.blank_id,
            T,
            num_states,
            num_forward_arcs,
            num_unit_arcs,
            forward_arcs_from_states,
            forward_arcs_to_states,
            unit_arcs_from_states,
            unit_arcs_to_states,
        )
        # topological sort
        arcs = self.top_sort(arcs, U, T)

        T_index = torch.div(arcs[:, 0], (U + 1), rounding_mode="floor")
        T_index[-1] = -1
        U_index = arcs[:, 0] % (U + 1)
        U_index[-1] = -1
        # sort by from state (required by k2)
        arcs, T_index, U_index = self.from_state_sort(arcs, T_index, U_index)
        print(arcs)

        transducer_graph = k2.Fsa(arcs, T_index)
        transducer_graph.u_index = U_index
        return transducer_graph

    def get_forward_arcs_states(
        self,
        num_forward_arcs: int,
        U: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from_states = torch.arange(num_forward_arcs, device=self.device)
        to_states = from_states + U + 1
        return from_states, to_states

    def get_unit_arcs_states(
        self,
        num_states: int,
        U: int,
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from_states = (
            torch.arange(num_states, device=self.device)
            .reshape(T, U + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        return from_states, to_states

    def assemble_arcs(
        self,
        unit_ids: torch.Tensor,
        blank_id: int,
        T: int,
        num_states: int,
        num_forward_arcs: int,
        num_unit_arcs: int,
        forward_arcs_from_states: torch.Tensor,
        forward_arcs_to_states: torch.Tensor,
        unit_arcs_from_states: torch.Tensor,
        unit_arcs_to_states: torch.Tensor,
    ) -> torch.Tensor:
        arcs = torch.zeros(
            (num_forward_arcs + num_unit_arcs + 2, 4),
            dtype=torch.int32,
            device=self.device,
        )
        arcs[:num_forward_arcs, 0] = forward_arcs_from_states
        arcs[:num_forward_arcs, 1] = forward_arcs_to_states
        arcs[:num_forward_arcs, 2] = blank_id

        ilabels = unit_ids.expand(T, -1).flatten()
        arcs[num_forward_arcs:-2, 0] = unit_arcs_from_states
        arcs[num_forward_arcs:-2, 1] = unit_arcs_to_states
        arcs[num_forward_arcs:-2, 2] = ilabels

        arcs[-2, 0] = num_states - 1
        arcs[-2, 1] = num_states
        arcs[-2, 2] = blank_id
        # final state in k2
        arcs[-1, 0] = num_states
        arcs[-1, 1] = num_states + 1
        arcs[-1, 2] = -1

        return arcs

    def top_sort(
        self,
        arcs: torch.Tensor,
        U: int,
        T: int,
    ) -> torch.Tensor:
        arcs[:-2, 0] = self.top_sort_states(arcs[:-2, 0], U + 1, T)
        arcs[:-3, 1] = self.top_sort_states(arcs[:-3, 1], U + 1, T)
        return arcs

    # TODO (Dongji): rewrite this funciton to be more readable
    def top_sort_states(
        self,
        states: torch.Tensor,
        n: int,
        m: int,
    ) -> torch.Tensor:
        """
        This function is from Nemo:
        (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/k2/graph_transducer.py)

        Relabel states to be in topological order: by diagonals

        Args:
            states: tensor with states
            n: number of rows
            m: number of columns

        Returns:
            tensor with relabeled states (same shape as `states`)
        """
        i = states % n
        j = torch.div(
            states, n, rounding_mode="floor"
        )  # states // n, torch.div to avoid pytorch warnings
        min_mn = min(m, n)
        max_mn = max(m, n)
        diag = i + j
        anti_diag = m + n - 1 - diag
        max_idx = n * m - 1
        cur_diag_idx = i if m > n else m - j - 1
        states = (
            diag.lt(min_mn) * ((diag * (diag + 1) >> 1) + i)
            + torch.logical_and(diag.ge(min_mn), diag.lt(max_mn))
            * ((min_mn * (min_mn + 1) >> 1) + (diag - min_mn) * min_mn + cur_diag_idx)
            + diag.ge(max_mn) * (max_idx - (anti_diag * (anti_diag + 1) >> 1) + m - j)
        )
        return states

    def from_state_sort(
        self,
        arcs: torch.Tensor,
        T_index: torch.Tensor,
        U_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, indices = torch.sort(arcs[:, 0], dim=0)
        arcs = arcs[indices]
        T_index = T_index[indices]
        U_index = U_index[indices]
        return arcs, T_index, U_index

    def texts_to_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.sp.encode(texts, out_type=int)
