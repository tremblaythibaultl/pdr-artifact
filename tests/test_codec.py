import numpy as np
import pytest

from pdr.codec import build_D, decode_doc, encode_doc


def test_roundtrip_ascii():
    text = "hello world"
    row = encode_doc(text, ell=32)
    assert row.shape == (32,)
    assert decode_doc(row) == text


def test_roundtrip_utf8():
    text = "café — résumé — π"
    ell = len(text.encode("utf-8")) + 4
    row = encode_doc(text, ell=ell)
    assert decode_doc(row) == text


def test_overflow_raises():
    with pytest.raises(ValueError):
        encode_doc("xxxxx", ell=3)


def test_build_D_pads_to_max():
    docs = ["a", "bb", "ccc"]
    D, ell = build_D(docs)
    assert D.shape == (3, 3)
    assert ell == 3
    assert decode_doc(D[0]) == "a"
    assert decode_doc(D[1]) == "bb"
    assert decode_doc(D[2]) == "ccc"


def test_build_D_empty_doc_decodes_empty():
    D, _ = build_D(["", "ab"])
    assert decode_doc(D[0]) == ""
    assert decode_doc(D[1]) == "ab"
