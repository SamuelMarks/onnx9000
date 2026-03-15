"""Module providing core logic and structural definitions."""

import pytest
import math
from onnx9000.extensions.audio.stft import (
    hann_window,
    hamming_window,
    dft,
    fft,
    stft,
    power_spectrogram,
    mel_filterbank,
    mel_spectrogram,
    log_mel_spectrogram,
)


def test_windows():
    """Provides semantic functionality and verification."""
    hann = hann_window(3)
    assert math.isclose(hann[0], 0.0, abs_tol=1e-05)
    assert math.isclose(hann[1], 1.0, abs_tol=1e-05)
    assert math.isclose(hann[2], 0.0, abs_tol=1e-05)
    hamm = hamming_window(3)
    assert math.isclose(hamm[0], 0.08, abs_tol=1e-05)
    assert math.isclose(hamm[1], 1.0, abs_tol=1e-05)
    assert math.isclose(hamm[2], 0.08, abs_tol=1e-05)


def test_dft_fft():
    """Provides semantic functionality and verification."""
    x = [1.0, 2.0, 3.0, 4.0]
    d_out = dft(x)
    f_out = fft(x)
    for i in range(4):
        assert math.isclose(d_out[i][0], f_out[i][0], abs_tol=1e-05)
        assert math.isclose(d_out[i][1], f_out[i][1], abs_tol=1e-05)
    assert math.isclose(f_out[0][0], 10.0, abs_tol=1e-05)
    assert fft([]) == []
    assert fft([5.0]) == [(5.0, 0.0)]


def test_stft():
    """Provides semantic functionality and verification."""
    waveform = [1.0, 1.0, 1.0, 1.0]
    spec = stft(waveform, n_fft=2, hop_length=1, win_length=2, center=False)
    assert len(spec) == 2
    assert len(spec[0]) == 3
    spec2 = stft(waveform, n_fft=2, hop_length=1, win_length=2, center=True)
    assert len(spec2) == 2
    assert len(spec2[0]) == 5
    spec3 = stft(waveform, n_fft=3, hop_length=1, win_length=3, center=False)
    assert len(spec3) == 3 // 2 + 1


def test_power_spectrogram():
    """Provides semantic functionality and verification."""
    waveform = [1.0, 1.0, 1.0, 1.0]
    spec = power_spectrogram(
        waveform, n_fft=2, hop_length=1, win_length=2, center=False
    )
    assert len(spec) == 2
    assert spec[0][0] >= 0


def test_mel_filterbank():
    """Provides semantic functionality and verification."""
    fbank = mel_filterbank(n_freqs=5, n_mels=2, sample_rate=16000)
    assert len(fbank) == 2
    assert len(fbank[0]) == 5


def test_mel_spectrogram():
    """Provides semantic functionality and verification."""
    waveform = [0.1] * 10
    mel_spec = mel_spectrogram(
        waveform, sample_rate=16000, n_fft=4, hop_length=2, win_length=4, n_mels=2
    )
    assert len(mel_spec) == 2
    assert len(mel_spec[0]) == 6


def test_log_mel_spectrogram():
    """Provides semantic functionality and verification."""
    mel_spec = [[0.0, 1.0, 10.0]]
    log_spec = log_mel_spectrogram(mel_spec, eps=1e-10)
    assert math.isclose(log_spec[0][0], math.log10(1e-10) * 10.0, abs_tol=1e-05)
    assert math.isclose(log_spec[0][1], 0.0, abs_tol=1e-05)
    assert math.isclose(log_spec[0][2], 10.0, abs_tol=1e-05)
