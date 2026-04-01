import pytest

from nanodiffusion.tokenizer import SpecialToken, Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


def test_encode_decode_roundtrip(tok: Tokenizer) -> None:
    text = "Hello, world! This is a test."
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unicode(tok: Tokenizer) -> None:
    text = "café naïve résumé"
    assert tok.decode(tok.encode(text)) == text


def test_mask_token_id_equals_base_vocab_size(tok: Tokenizer) -> None:
    assert tok.mask_token_id == tok.base_vocab_size


def test_vocab_size_includes_all_special_tokens(tok: Tokenizer) -> None:
    assert tok.vocab_size == tok.base_vocab_size + len(SpecialToken)


def test_special_token_ids_are_unique(tok: Tokenizer) -> None:
    ids = [tok.encode_special(t) for t in SpecialToken]
    assert len(ids) == len(set(ids))


def test_special_token_ids_above_base_vocab(tok: Tokenizer) -> None:
    for t in SpecialToken:
        assert tok.encode_special(t) >= tok.base_vocab_size


def test_decode_mask_token(tok: Tokenizer) -> None:
    ids = [*tok.encode("hello"), tok.mask_token_id, *tok.encode(" world")]
    assert tok.decode(ids) == "hello<|mask|> world"


def test_encode_special_via_enum(tok: Tokenizer) -> None:
    assert tok.encode_special(SpecialToken.MASK) == tok.mask_token_id
    assert tok.encode_special(SpecialToken.BOS) == tok.bos_token_id
    assert tok.encode_special(SpecialToken.EOS) == tok.eos_token_id
