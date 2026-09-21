from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pdr.codec import decode_doc
from pdr.matvec import MatVecBackend, PlaintextBackend


@dataclass
class Server:
    A: np.ndarray
    D: np.ndarray
    backend: MatVecBackend

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def ell(self) -> int:
        return int(self.D.shape[1])

    def round1(self, c: np.ndarray) -> np.ndarray:
        return self.backend.matvec(self.A, c)

    def round2(self, c: np.ndarray) -> np.ndarray:
        return self.backend.matvec(self.D.T, c)


@dataclass
class Client:
    def round1_query(self, x_int: np.ndarray) -> np.ndarray:
        return x_int.astype(np.int64, copy=False)

    def round1_decode(self, scores: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        top = np.argsort(scores)[::-1][:k]
        return scores, top

    def round2_query(self, i_star: int, m: int) -> np.ndarray:
        e = np.zeros(m, dtype=np.int64)
        e[i_star] = 1
        return e

    def round2_decode(self, row: np.ndarray) -> str:
        return decode_doc(row)


def run_protocol(client: Client, server: Server, x_int: np.ndarray) -> tuple[int, str, np.ndarray]:
    c1 = client.round1_query(x_int)
    s = server.round1(c1)
    scores, top = client.round1_decode(s, k=1)
    i_star = int(top[0])
    c2 = client.round2_query(i_star, server.m)
    row = server.round2(c2)
    text = client.round2_decode(row)
    return i_star, text, scores


def make_server(A: np.ndarray, D: np.ndarray, backend: MatVecBackend | None = None) -> Server:
    backend = backend or PlaintextBackend()
    return Server(A=A.astype(np.int64), D=D.astype(np.int64), backend=backend)
