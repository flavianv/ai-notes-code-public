# A1: exact facts used in the script, and sanity checks on the attention-only model.
import torch
from atlib.circuits import AttnOnly, repeat_batch, pair_detector_mlp
from atlib.automata import TRANSITIONS


def test_softmax_example():
    assert round(torch.tensor([0., 6., 0., 0.]).softmax(-1)[1].item() * 100, 2) == 99.26


def test_pair_detector_is_exact():
    f = pair_detector_mlp(TRANSITIONS); oh = lambda i: torch.eye(3)[i]
    for s in range(3):
        for ai, a in enumerate("ABI"):
            out = f(oh(s), oh(ai))
            assert out.tolist() == oh(TRANSITIONS[a][s]).tolist()     # exactly one-hot on the next state, all 9 pairs


def test_repeat_data_and_head_switch():
    ids = repeat_batch(4, half=5, gen=torch.Generator().manual_seed(0))
    assert ids.shape == (4, 11) and (ids[:, 1:6] == ids[:, 6:]).all()
    torch.manual_seed(0); m = AttnOnly(50).eval()
    with torch.no_grad():
        a, _ = m(ids); b, _ = m(ids, off={(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)})
        c, _ = m(ids, off={(1, 2)})
    assert (a - b).abs().max() > 1e-3 and (a - c).abs().max() > 1e-4     # switching heads off changes the output
