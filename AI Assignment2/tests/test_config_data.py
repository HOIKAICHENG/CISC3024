import code.config as config_module
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


def test_run_metadata_reports_effective_quick_and_full_policies():
    config = ExperimentConfig(
        map_epochs=7,
        abnn_epochs=4,
        member_seeds=(11, 12, 13),
        mc_samples=9,
    )

    quick = config_module.run_metadata(config, quick=True)
    full = config_module.run_metadata(config, quick=False)

    assert quick == {
        "mode": "quick",
        "quick": True,
        "effective": {
            "dataset_limits": {
                "train": 1000,
                "cifar10_test": 500,
                "svhn_test": 500,
                "cifar100_test": 500,
            },
            "map_epochs": 1,
            "abnn_epochs": 1,
            "member_seeds": [11],
            "member_count": 1,
            "mc_samples": 2,
        },
    }
    assert full == {
        "mode": "full",
        "quick": False,
        "effective": {
            "dataset_limits": {
                "train": None,
                "cifar10_test": None,
                "svhn_test": None,
                "cifar100_test": None,
            },
            "map_epochs": 7,
            "abnn_epochs": 4,
            "member_seeds": [11, 12, 13],
            "member_count": 3,
            "mc_samples": 3,
        },
    }
