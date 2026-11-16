import sys
import torch

sys.path.insert(0, ".")
from scripts.train_mechanism import _run_with_early_stopping


def _mock_training_loop(val_sequence, patience, min_epochs, max_epochs=80):
    """Drives _run_with_early_stopping with a scripted validation sequence
    (one value per epoch, extended by repeating the last value if
    max_epochs exceeds the sequence) and no-op training/state, to test
    the stopping LOGIC in isolation from real training."""
    calls = {"epoch": 0}

    def train_one_epoch_fn(idx):
        pass

    def val_eval_fn():
        i = min(calls["epoch"], len(val_sequence) - 1)
        calls["epoch"] += 1
        return val_sequence[i]

    state_modules = {"m": torch.nn.Linear(2, 2)}
    actual_epochs = _run_with_early_stopping(
        max_epochs, patience, torch.arange(4), torch.device("cpu"), batch_size=4,
        train_one_epoch_fn=train_one_epoch_fn, val_eval_fn=val_eval_fn, state_modules=state_modules,
        min_epochs=min_epochs)
    return actual_epochs


def test_plateau_then_breakout_does_not_stop_prematurely():
    """12 flat epochs (the GeoLife cold-start length) then real
    improvement -- with min_epochs=20 this must NOT stop during the
    plateau (patience=8 would otherwise fire at epoch 9: 1 to set the
    baseline + 8 flat epochs)."""
    val_seq = [0.06] * 12 + [0.10, 0.15, 0.20, 0.25, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
    actual_epochs = _mock_training_loop(val_seq, patience=8, min_epochs=20)
    assert actual_epochs > 12, f"stopped at epoch {actual_epochs}, during or before the plateau ended"
    print(f"OK: 12-epoch plateau then breakout -> training ran {actual_epochs} epochs "
          f"(did not stop prematurely during the plateau)")


def test_genuine_early_convergence_is_not_penalized():
    """Real improvement for a few epochs then a genuine plateau should
    still stop once min_epochs is cleared and patience is exceeded,
    not run needlessly to max_epochs."""
    val_seq = [0.1, 0.3, 0.5, 0.6, 0.65] + [0.65] * 30
    actual_epochs = _mock_training_loop(val_seq, patience=8, min_epochs=20)
    assert actual_epochs < 80, "ran to max_epochs instead of stopping after genuine convergence"
    assert actual_epochs >= 20, "stopped before min_epochs, which should never happen"
    print(f"OK: genuine early convergence still stops at epoch {actual_epochs} (not forced to max_epochs)")


def test_without_burnin_old_behavior_would_have_stopped_in_plateau():
    """Sanity check that the scenario in test 1 WOULD have stopped
    prematurely with min_epochs=0 -- confirms the test is actually
    exercising the plateau scenario min_epochs is meant to guard
    against, not a scenario that was already safe."""
    val_seq = [0.06] * 12 + [0.10, 0.15, 0.20, 0.25, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
    actual_epochs = _mock_training_loop(val_seq, patience=8, min_epochs=0)
    assert actual_epochs <= 12, (
        f"expected min_epochs=0 to stop during the plateau (<=12 epochs), "
        f"got {actual_epochs} -- the test scenario no longer reproduces the plateau")
    print(f"OK: confirmed min_epochs=0 stops at epoch {actual_epochs}, "
          f"during the plateau -- this is exactly what min_epochs=20 prevents")


if __name__ == "__main__":
    test_plateau_then_breakout_does_not_stop_prematurely()
    test_genuine_early_convergence_is_not_penalized()
    test_without_burnin_old_behavior_would_have_stopped_in_plateau()
    print("All min_epochs burn-in tests passed.")
