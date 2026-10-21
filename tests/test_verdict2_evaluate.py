"""Offline regression tests for Verdict 2.0 checkpoint preflight."""

import pickle
import sys

import pytest
import torch

from verdict2 import evaluate


@pytest.fixture
def checkpoint_path(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verdict2.evaluate",
            "--checkpoint", str(path),
            "--out", str(tmp_path / "report.json"),
            "--device", "cpu",
            "--confirm",
        ],
    )
    return path


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_lfs_pointer_rejected_before_torch_load(checkpoint_path, monkeypatch, newline):
    checkpoint_path.write_bytes(newline.join([
        b"version https://git-lfs.github.com/spec/v1",
        b"oid sha256:" + b"0" * 64,
        b"size 598509338",
        b"",
    ]))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Git LFS pointers must be rejected before torch.load")

    monkeypatch.setattr(evaluate.torch, "load", unexpected_load)

    with pytest.raises(SystemExit) as exc:
        evaluate.main()

    message = str(exc.value)
    assert str(checkpoint_path) in message
    assert "Git LFS pointer" in message
    assert "not model weights" in message
    assert "git lfs pull" in message
    assert "404" in message
    assert "maintainer" in message
    assert not (checkpoint_path.parent / "report.json").exists()


@pytest.mark.parametrize("use_zip", [True, False])
def test_non_pointer_checkpoint_uses_existing_loader(checkpoint_path, monkeypatch, use_zip):
    torch.save(
        {"backbone": "offline-test", "state_dict": {"weight": torch.tensor([1.0])}},
        checkpoint_path,
        _use_new_zipfile_serialization=use_zip,
    )
    original_load = torch.load

    class CheckpointLoaded(Exception):
        pass

    def load_and_stop(path, **kwargs):
        assert path == str(checkpoint_path)
        assert kwargs == {"map_location": "cpu", "weights_only": False}
        checkpoint = original_load(path, **kwargs)
        assert checkpoint["backbone"] == "offline-test"
        assert torch.equal(checkpoint["state_dict"]["weight"], torch.tensor([1.0]))
        raise CheckpointLoaded

    monkeypatch.setattr(evaluate.torch, "load", load_and_stop)

    with pytest.raises(CheckpointLoaded):
        evaluate.main()


def test_missing_checkpoint_preserves_file_error(checkpoint_path):
    with pytest.raises(FileNotFoundError) as exc:
        evaluate.main()
    assert exc.value.filename == str(checkpoint_path)


def test_corrupt_non_pointer_preserves_load_error(checkpoint_path):
    checkpoint_path.write_bytes(b"not a checkpoint")
    with pytest.raises(pickle.UnpicklingError):
        evaluate.main()


def test_confirmation_required_before_checkpoint_read(checkpoint_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])

    with pytest.raises(SystemExit) as exc:
        evaluate.main()

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Anti-leak checklist" in output
    for line in evaluate.CHECKLIST:
        assert line in output
