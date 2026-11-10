import numpy as np
import torch

from fingerspell import evaluate, train, validate
from fingerspell.data import Samples
from fingerspell.decoder import LetterDecoder
from fingerspell.features import normalize
from fingerspell.model import augment
from fingerspell.splits import group_split
from fingerspell.synthetic import make_synthetic


def test_normalize_ignores_position_and_size():
    rng = np.random.default_rng(0)
    hand = rng.normal(0, 50, size=(1, 21, 3)).astype(np.float32)
    moved = hand * 3.0
    moved[..., :2] += [200, -75]
    np.testing.assert_allclose(normalize(hand, [0]), normalize(moved, [0]), atol=1e-4)


def test_left_hand_maps_onto_right():
    rng = np.random.default_rng(1)
    right = rng.normal(0, 50, size=(1, 21, 3)).astype(np.float32)
    left = right.copy()
    left[..., 0] *= -1
    np.testing.assert_allclose(normalize(right, [0]), normalize(left, [1]), atol=1e-4)


def test_normalize_keeps_orientation():
    rng = np.random.default_rng(2)
    hand = rng.normal(0, 50, size=(1, 21, 3)).astype(np.float32)
    turned = hand[..., [1, 0, 2]] * [-1, 1, 1]  # 90 degree in-plane rotation
    assert np.abs(normalize(hand, [0]) - normalize(turned.astype(np.float32), [0])).max() > 0.1


def test_split_keeps_groups_together_and_covers_classes():
    s = make_synthetic(classes=("A", "B", "C"), per_class=200)
    split = group_split(s.labels, s.group, seed=0)
    for g in np.unique(s.group):
        assert len(set(split[s.group == g])) == 1
    for name in ("train", "val", "test"):
        assert set(s.labels[split == name]) == {"A", "B", "C"}


def test_augment_preserves_shape_and_stays_close():
    x = torch.randn(16, 63)
    out = augment(x)
    assert out.shape == x.shape
    assert (out - x).abs().mean() < 0.5


def test_samples_roundtrip(tmp_path):
    s = make_synthetic(classes=("A", "B"), per_class=10)
    s.save(tmp_path / "d.npz")
    back = Samples.load(tmp_path / "d.npz")
    np.testing.assert_array_equal(back.landmarks, s.landmarks)
    assert back.labels.tolist() == s.labels.tolist()


def one_hot(i, n=3):
    p = np.full(n, 0.01)
    p[i] = 1 - 0.01 * (n - 1)
    return p


def test_decoder_commits_once_per_hold():
    d = LetterDecoder(["A", "B", "space"], hold_frames=5, release_frames=3)
    for _ in range(40):
        d.update(one_hot(0))
    assert d.text == "A"
    for _ in range(20):
        d.update(one_hot(1))
    assert d.text == "AB"
    for _ in range(3):
        d.update(None)
    for _ in range(10):
        d.update(one_hot(1))
    assert d.text == "ABB"
    for _ in range(20):
        d.update(one_hot(2))
    assert d.text == "ABB "


def test_decoder_ignores_brief_and_uncertain_frames():
    d = LetterDecoder(["A", "B", "C"], hold_frames=5)
    for _ in range(3):
        d.update(one_hot(0))
    for _ in range(30):
        d.update(np.array([0.4, 0.3, 0.3]))
    assert d.text == ""
    # a confidence dip in the middle of a held letter must not type it twice
    d = LetterDecoder(["A", "B", "C"], hold_frames=5)
    for _ in range(15):
        d.update(one_hot(0))
    for _ in range(6):
        d.update(np.array([0.4, 0.3, 0.3]))
    for _ in range(15):
        d.update(one_hot(0))
    assert d.text == "A"


def test_train_evaluate_validate_end_to_end(tmp_path):
    make_synthetic(per_class=120, signer="s1").save(tmp_path / "s1.npz")
    make_synthetic(per_class=40, signer="s2", seed=0).save(tmp_path / "s2.npz")
    run = tmp_path / "run"
    summary = train.main(["--data", str(tmp_path / "s1.npz"), str(tmp_path / "s2.npz"),
                          "--out", str(run), "--epochs", "8", "--holdout-signer", "s2"])
    assert summary["n_test"] == 26 * 40
    metrics = evaluate.main(["--run", str(run)])
    assert metrics["accuracy"] > 0.95
    assert list(metrics["by_signer"]) == ["s2"]
    assert (run / "confusion_test.png").exists()
    assert validate.main([str(tmp_path / "s1.npz")]) == 0
