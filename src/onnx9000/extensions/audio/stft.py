"""Module providing core logic and structural definitions."""

import math
from typing import List, Tuple


def hann_window(window_length: int) -> List[float]:
    """Generates a Hann window."""
    return [
        0.5 - 0.5 * math.cos(2 * math.pi * n / (window_length - 1))
        for n in range(window_length)
    ]


def hamming_window(window_length: int) -> List[float]:
    """Generates a Hamming window."""
    return [
        0.54 - 0.46 * math.cos(2 * math.pi * n / (window_length - 1))
        for n in range(window_length)
    ]


def dft(x: List[float]) -> List[Tuple[float, float]]:
    """
    Discrete Fourier Transform.
    Returns a list of complex numbers represented as (real, imag) tuples.
    """
    N = len(x)
    out = []
    for k in range(N):
        real = 0.0
        imag = 0.0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            real += x[n] * math.cos(angle)
            imag -= x[n] * math.sin(angle)
        out.append((real, imag))
    return out


def fft(x: List[float]) -> List[Tuple[float, float]]:
    """
    Cooley-Tukey Radix-2 Fast Fourier Transform.
    x must have a power-of-2 length.
    """
    N = len(x)
    if N <= 1:
        return [(val, 0.0) for val in x]

    even = fft(x[0::2])
    odd = fft(x[1::2])

    out = [(0.0, 0.0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        odd_k_real, odd_k_imag = odd[k]

        t_real = cos_a * odd_k_real - sin_a * odd_k_imag
        t_imag = sin_a * odd_k_real + cos_a * odd_k_imag

        even_k_real, even_k_imag = even[k]

        out[k] = (even_k_real + t_real, even_k_imag + t_imag)
        out[k + N // 2] = (even_k_real - t_real, even_k_imag - t_imag)

    return out


def stft(
    waveform: List[float],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: List[float] = None,
    center: bool = True,
) -> List[List[Tuple[float, float]]]:
    """
    Short-Time Fourier Transform.
    """
    if window is None:
        window = hann_window(win_length)

    if center:
        pad_amount = n_fft // 2
        # Reflect padding
        left_pad = waveform[1 : pad_amount + 1][::-1]
        right_pad = waveform[-pad_amount - 1 : -1][::-1]
        padded_waveform = left_pad + waveform + right_pad
    else:
        padded_waveform = waveform

    num_frames = 1 + (len(padded_waveform) - n_fft) // hop_length
    out = []

    is_power_of_2 = (n_fft != 0) and ((n_fft & (n_fft - 1)) == 0)

    for i in range(num_frames):
        start = i * hop_length
        frame = padded_waveform[start : start + n_fft]

        # Apply window
        windowed_frame = [0.0] * n_fft
        # Centering the window inside n_fft if win_length < n_fft
        pad_left = (n_fft - win_length) // 2
        for j in range(win_length):
            windowed_frame[pad_left + j] = frame[pad_left + j] * window[j]

        if is_power_of_2:
            spec = fft(windowed_frame)
        else:
            spec = dft(windowed_frame)

        # Return only the non-redundant part (up to Nyquist frequency)
        out.append(spec[: n_fft // 2 + 1])

    # Transpose to shape (Freq, Frames) to match standard PyTorch behavior usually,
    # but returning (Frames, Freq) here as list of frames.
    # PyTorch returns (Freq, Frames). Let's do (Freq, Frames)
    freq_bins = n_fft // 2 + 1
    transposed = []
    for f in range(freq_bins):
        row = []
        for t in range(num_frames):
            row.append(out[t][f])
        transposed.append(row)

    return transposed


def power_spectrogram(
    waveform: List[float],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: List[float] = None,
    center: bool = True,
) -> List[List[float]]:
    """
    Calculates the power spectrogram (magnitude squared).
    """
    spec = stft(waveform, n_fft, hop_length, win_length, window, center)
    out = []
    for row in spec:
        p_row = []
        for real, imag in row:
            p_row.append(real * real + imag * imag)
        out.append(p_row)
    return out


def hz_to_mel(freq: float) -> float:
    """Provides semantic functionality and verification."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: float) -> float:
    """Provides semantic functionality and verification."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_freqs: int, n_mels: int, sample_rate: int, f_min: float = 0.0, f_max: float = None
) -> List[List[float]]:
    """
    Generates a Mel-filterbank matrix.
    Shape: (n_mels, n_freqs)
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    min_mel = hz_to_mel(f_min)
    max_mel = hz_to_mel(f_max)

    mels = [min_mel + i * (max_mel - min_mel) / (n_mels + 1) for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]

    # bin indices
    bins = [math.floor((n_freqs - 1) * h / (sample_rate / 2.0)) for h in hzs]

    fbank = [[0.0] * n_freqs for _ in range(n_mels)]

    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]

        for j in range(left, center):
            fbank[i][j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i][j] = (right - j) / (right - center)

    return fbank


def mel_spectrogram(
    waveform: List[float],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float = None,
) -> List[List[float]]:
    """Provides semantic functionality and verification."""
    power_spec = power_spectrogram(waveform, n_fft, hop_length, win_length, center=True)
    n_freqs = len(power_spec)

    fbank = mel_filterbank(n_freqs, n_mels, sample_rate, f_min, f_max)

    num_frames = len(power_spec[0])
    out = [[0.0] * num_frames for _ in range(n_mels)]

    for m in range(n_mels):
        for t in range(num_frames):
            val = 0.0
            for f in range(n_freqs):
                val += fbank[m][f] * power_spec[f][t]
            out[m][t] = val

    return out


def log_mel_spectrogram(
    mel_spec: List[List[float]], eps: float = 1e-10
) -> List[List[float]]:
    """Provides semantic functionality and verification."""
    out = []
    for row in mel_spec:
        out.append([math.log10(max(eps, val)) * 10.0 for val in row])
    return out
