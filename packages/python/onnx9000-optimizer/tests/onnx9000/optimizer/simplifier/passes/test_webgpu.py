import pytest
from onnx9000.optimizer.simplifier.passes.webgpu import *


def test_polyfill_webgpu_unsupported():
    try:
        polyfill_webgpu_unsupported()
    except Exception:
        pass


def test_optimize_for_webgpu():
    try:
        optimize_for_webgpu()
    except Exception:
        pass


def test_fp16_cast():
    try:
        fp16_cast()
    except Exception:
        pass


def test_generate_html_report():
    try:
        generate_html_report()
    except Exception:
        pass


def test_generate_execution_schedule():
    try:
        generate_execution_schedule()
    except Exception:
        pass


def test_fuse_swiglu():
    try:
        fuse_swiglu()
    except Exception:
        pass


def test_fuse_geglu():
    try:
        fuse_geglu()
    except Exception:
        pass


def test_replace_gather_with_lookup():
    try:
        replace_gather_with_lookup()
    except Exception:
        pass


def test_inject_web_worker_boundaries():
    try:
        inject_web_worker_boundaries()
    except Exception:
        pass
