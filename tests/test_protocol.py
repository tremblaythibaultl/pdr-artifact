import numpy as np

from pdr.codec import build_D
from pdr.protocol import Client, make_server, run_protocol
from pdr.quantize import quantize


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def test_protocol_preserves_argmax_on_synthetic():
    rng = np.random.default_rng(7)
    m, n, q = 100, 32, 50
    bits = 8

    A_float = _normalize(rng.standard_normal((m, n)).astype(np.float32))
    targets = rng.integers(0, m, size=q)
    noise = 0.15 * rng.standard_normal((q, n)).astype(np.float32)
    Q_float = _normalize(A_float[targets] + noise)

    A_int, sA = quantize(A_float, bits=bits)
    docs = [f"doc-{i}: " + ("x" * (i % 7 + 1)) for i in range(m)]
    D_int, _ = build_D(docs)

    server = make_server(A_int, D_int)
    client = Client()

    matches = 0
    for x_float in Q_float:
        x_int, _ = quantize(x_float, bits=bits, scale=sA)
        i_star_protocol, text, _ = run_protocol(client, server, x_int)
        i_star_float = int(np.argmax(A_float @ x_float))
        if i_star_protocol == i_star_float:
            matches += 1
        assert text == docs[i_star_protocol]
    assert matches / q >= 0.95


def test_protocol_recovers_document_text():
    rng = np.random.default_rng(11)
    m, n = 20, 16
    A_int = rng.integers(-3, 4, size=(m, n), dtype=np.int64)
    docs = [f"document number {i} — résumé" for i in range(m)]
    D_int, _ = build_D(docs)

    server = make_server(A_int, D_int)
    client = Client()

    x_int = A_int[7]
    i_star, text, _ = run_protocol(client, server, x_int)
    assert i_star == 7
    assert text == docs[7]


def test_round1_matches_direct_matvec():
    rng = np.random.default_rng(13)
    m, n = 40, 32
    A_int = rng.integers(-30, 31, size=(m, n), dtype=np.int64)
    x_int = rng.integers(-30, 31, size=n, dtype=np.int64)
    D_int = np.zeros((m, 1), dtype=np.int64)

    server = make_server(A_int, D_int)
    client = Client()

    c1 = client.round1_query(x_int)
    scores = server.round1(c1)
    assert np.array_equal(scores, A_int @ x_int)
