# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2024  Johns Hopkins University (author: Dongji Gao)
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
Note we use `rnnt_loss` from torchaudio, which exists only in
torchaudio >= v0.10.0. It also means you have to use torch >= v1.10.0
"""
import k2
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional
from encoder_interface import EncoderInterface

from icefall.transducer_graph_compiler import TransducerTrainingGraphCompiler
from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        graph_compiler: TransducerTrainingGraphCompiler,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
          graph_compiler:
            It is used to compile the training graphs for the transducer
            model.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.graph_compiler = graph_compiler

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
        Returns:
          Return the transducer loss.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes
        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)
        decoder_out, _ = self.decoder(sos_y_padded)

        logits = self.joiner(encoder_out, decoder_out)

        training_graphs = self.graph_compiler.compile(y, x_lens)
        lattice = self.make_lattice(
            training_graphs,
            logits,
        )

        loss = -1 * lattice.get_tot_scores(
            use_double_scores=True,
            log_semiring=True,
        )

        return loss


def make_lattice(
    self,
    training_graphs: k2.Fsa,
    logits: torch.Tensor,
):
    """Assign scores to the given graph using the given logits.

    Args:
      graph:
        It is the training graph. It must have the attribute `aux_labels`.
      logits:
        It is a 4-D tensor of shape (N, T, U + 1, C). It is the output
        of the joiner.

    Returns:
      Return a new FsaVec with scores assigned.
    """
    device = training_graphs.device
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    N, T, U, C = log_probs.shape
    batch_offset = T * U * C
    time_offset = U * C
    unit_offset = C
    with torch.no_grad():
        batch_indices = torch.repeat_interleave(
            torch.tensor(
                [training_graphs[i].num_arcs for i in range(N)],
                device=device,
            )
        )
        score_indices = (
            batch_indices * batch_offset
            + training_graphs.aux_labels.to(torch.int64) * time_offset
            + training_graphs.u_index.to(torch.int64) * unit_offset
            + training_graphs.labels.to(torch.int64)
        )
        score_indices[training_graphs.labels == -1] = 0
    scores = log_probs.flatten().index_select(-1, score_indices)
    scores[training_graphs.labels == -1] = 0

    training_graphs.scores = scores

    return training_graphs
