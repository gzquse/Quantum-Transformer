from pathlib import Path
import torch
import numpy as np
import sys, os
# Import QCrankEncoder from your module
from .ParametricQCrankV2 import ParametricQCrankV2
from qiskit import transpile
from qiskit_aer import AerSimulator

class QCrankEncoder:
    """Runs one ParametricQCrankV2 circuit per token."""

    def __init__(
        self,
        nq_addr: int = 2,
        nq_data: int = 2,
        shots: int = 2048,
    ):
        self.nq_addr = nq_addr
        self.nq_data = nq_data
        self.num_addr = 2 ** nq_addr
        self.out_dim = self.num_addr * self.nq_data
        self.shots = shots
        self.qcr = ParametricQCrankV2(
            nq_addr, nq_data, useCZ=False, measure=True, barrier=True
        ) # type: ignore
        self.backend = AerSimulator()

    # --------------------------------------------
    def _vec_to_matrix(self, vec: np.ndarray) -> np.ndarray:
        data = np.zeros(self.out_dim)
        L = min(len(vec), self.out_dim)
        data[:L] = np.clip(vec[:L], -1.0, 1.0)
        return data.reshape(self.num_addr, self.nq_data)

    def _encode_single(self, vec: np.ndarray) -> np.ndarray:
        mat = self._vec_to_matrix(vec)[..., np.newaxis]
        self.qcr.bind_data(mat)
        circ = self.qcr.instantiate_circuits()[0]
        counts = (
            self.backend.run(transpile(circ, self.backend), shots=self.shots)
            .result()
            .get_counts()
        )
        rec, _ = self.qcr.reco_from_yields([counts])
        return rec[:, :, 0].reshape(-1)

    # --------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        out = np.empty((b, l, self.out_dim), dtype=np.float32)
        for bi in range(b):
            for ti in range(l):
                out[bi, ti] = self._encode_single(x[bi, ti].cpu().numpy())
        return torch.from_numpy(out).to(x.device)