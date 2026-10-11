from code.config import ExperimentConfig
from code.data import deterministic_subset_indices


def test_approved_full_defaults():
    cfg = ExperimentConfig()
    assert (cfg.map_epochs, cfg.abnn_epochs, cfg.batch_size) == (20, 5, 128)
    assert (cfg.member_seeds, cfg.mc_samples) == ((101, 102, 103), 3)


def test_subset_indices_are_seeded_and_bounded():
    a = deterministic_subset_indices(100, 12, seed=7)
    b = deterministic_subset_indices(100, 12, seed=7)
    assert a == b
    assert len(a) == len(set(a)) == 12
    assert min(a) >= 0 and max(a) < 100
