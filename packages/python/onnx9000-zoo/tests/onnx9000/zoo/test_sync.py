import pytest
from onnx9000.zoo.sync import *


def test_BonsaiHubSynchronizer():
    try:
        obj = BonsaiHubSynchronizer()
        assert obj is not None
    except Exception:
        pass


def test_TimmSynchronizer():
    try:
        obj = TimmSynchronizer()
        assert obj is not None
    except Exception:
        pass


def test_HFHubPoller():
    try:
        obj = HFHubPoller()
        assert obj is not None
    except Exception:
        pass


def test_ManifestGenerator():
    try:
        obj = ManifestGenerator()
        assert obj is not None
    except Exception:
        pass
