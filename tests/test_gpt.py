"""The from-scratch GPT: key/value-cached generation must match a full forward pass."""
import pytest
import torch

from model.gpt import GPT, GPTConfig, Penalties, sample


def tiny(block=32, vocab=97):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32)).eval()


def reference_generate(m, idx, n):
    """The previous implementation, greedy: the whole (cropped) context through the model per token."""
    for _ in range(n):
        logits, _ = m(idx[:, -m.config.block_size:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return idx


def test_cached_logits_match_full_forward():
    m = tiny()
    x = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        first, _ = m(x[:, :12], cache=cache)          # prompt
        assert cache.pos == 12
        torch.testing.assert_close(first[:, -1], full[:, 11], rtol=1e-4, atol=1e-4)
        for t in range(12, 20):                        # then one token at a time
            step, _ = m(x[:, t:t + 1], cache=cache)
            torch.testing.assert_close(step[:, -1], full[:, t], rtol=1e-4, atol=1e-4)


def test_cached_multi_token_append_matches_full_forward():
    m = tiny()
    x = torch.randint(0, 97, (1, 16))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        m(x[:, :10], cache=cache)
        out, _ = m(x[:, 10:16], cache=cache)           # several new tokens on top of a cache
    torch.testing.assert_close(out[:, -1], full[:, 15], rtol=1e-4, atol=1e-4)


def test_greedy_generate_matches_previous_implementation():
    m = tiny(block=64)
    x = torch.randint(0, 97, (1, 8))
    with torch.no_grad():
        want = reference_generate(m, x, 40)
    got = m.generate(x, 40, top_k=1)                   # top-k 1 = greedy
    assert torch.equal(got, want)


def test_generate_runs_past_the_context_window():
    m = tiny(block=16)
    x = torch.randint(0, 97, (1, 30))                  # prompt already longer than the window
    out = m.generate(x, 50)
    assert out.shape == (1, 80)
    assert torch.equal(out[:, :30], x)


def test_window_rebuild_keeps_the_last_tokens():
    m = tiny(block=16)
    x = torch.randint(0, 97, (1, 4))
    toks = [t.item() for t in m.stream(x, 40, top_k=1, keep=0.5)]
    assert len(toks) == 40


def test_stop_token_ends_generation():
    m = tiny()
    x = torch.randint(0, 97, (1, 5))
    first = m.generate(x, 1, top_k=1)[0, -1].item()
    out = m.generate(x, 20, top_k=1, stop_id=first)
    assert out.shape[1] == 6 and out[0, -1].item() == first


def test_sample_top_k_and_top_p():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 5.0, 1.0, 4.9]])
    assert all(sample(logits, 1.0, top_k=1, top_p=1.0).item() == 1 for _ in range(20))
    assert all(sample(logits, 1.0, top_k=0, top_p=0.01).item() == 1 for _ in range(20))
    picks = {sample(logits, 1.0, top_k=2, top_p=1.0).item() for _ in range(200)}
    assert picks == {1, 3}


def test_training_forward_is_unchanged_by_cache_code():
    m = tiny()
    x = torch.randint(0, 97, (2, 32))
    logits, loss = m(x, x)
    assert logits.shape == (2, 32, 97) and loss.ndim == 0
    loss.backward()
    assert m.wte.weight.grad is not None


@pytest.mark.parametrize("block", [8, 33])
def test_prompt_at_window_edges(block):
    m = tiny(block=block)
    x = torch.randint(0, 97, (1, block))
    assert m.generate(x, 3).shape == (1, block + 3)


# ---------------------------------------------------------------- anti-repetition


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def test_penalties_off_change_nothing():
    m = tiny(block=64)
    x = torch.randint(0, 97, (1, 8))
    assert not Penalties()
    assert torch.equal(m.generate(x, 30, top_k=1), m.generate(x, 30, top_k=1, penalties=Penalties()))


def test_greedy_tiny_model_loops_without_penalties_and_not_with_no_repeat_ngram():
    m = tiny(block=128)
    x = torch.randint(0, 97, (1, 4))
    plain = m.generate(x, 100, top_k=1)[0].tolist()
    assert len(set(ngrams(plain, 4))) < len(ngrams(plain, 4))  # an untrained model repeats itself
    fixed = m.generate(x, 100, top_k=1, penalties={"no_repeat_ngram": 4})[0].tolist()
    assert len(set(ngrams(fixed, 4))) == len(ngrams(fixed, 4))  # no 4 tokens ever twice


def test_no_repeat_ngram_survives_the_window_rebuild():
    m = tiny(block=32)
    x = torch.randint(0, 97, (1, 4))
    toks = [t.item() for t in m.stream(x, 80, top_k=1, keep=0.5, penalties={"no_repeat_ngram": 3})]
    assert len(toks) == 80


def test_repetition_presence_frequency_lower_used_tokens():
    logits = torch.tensor([[2.0, -1.0, 1.0, 0.5]])
    ctx = torch.tensor([[0, 1, 0]])
    out = Penalties(repetition=2.0).apply(logits, ctx)
    assert out.tolist() == [[1.0, -2.0, 1.0, 0.5]]  # positive halved, negative doubled, unused kept
    out = Penalties(presence=0.5, frequency=0.25).apply(logits, ctx)
    assert out.tolist() == [[2.0 - 0.5 - 0.5, -1.0 - 0.5 - 0.25, 1.0, 0.5]]


def test_no_repeat_ngram_bans_the_token_that_would_repeat():
    logits = torch.zeros(1, 5)
    ctx = torch.tensor([[1, 2, 3, 4, 1, 2]])  # "1 2" was followed by 3 before
    out = Penalties(no_repeat_ngram=3).apply(logits, ctx)
    assert out[0, 3] == -float("inf") and torch.isfinite(out[0, [0, 1, 2, 4]]).all()
    # a vocabulary where everything is banned falls back instead of failing
    out = Penalties(no_repeat_ngram=2).apply(torch.zeros(1, 2), torch.tensor([[0, 0, 1, 0]]))
    assert torch.isfinite(out).all()
