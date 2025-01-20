import torch
import numpy as np
import pytest
import librosa

from basic_pitch_torch.constants import AUDIO_SAMPLE_RATE
from basic_pitch_torch.inference import (
    load_basic_pitch_model,
    predict,
    predict_from_signal,
)


@pytest.fixture
def audio_path():
    return librosa.example("fishin")


@pytest.fixture
def signal(audio_path):
    sig, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    return torch.from_numpy(sig)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No CUDA device"
            ),
        ),
    ],
)
@pytest.mark.parametrize("signal_dims", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float32, 1e-6),
        (torch.bfloat16, 0.11),  # bfloat16 casting leads to large errors
    ],
)
def test_predict_from_signal(audio_path, signal, signal_dims, device, dtype, atol):
    # prepare signal for testing
    signal = signal.to(device=device, dtype=dtype)

    assert signal.ndim == 1
    while signal.ndim < signal_dims:
        signal = signal.unsqueeze(0)

    # check that predict_from_signal returns same results as predict
    model_output, _, _ = predict(audio_path)

    model = load_basic_pitch_model().to(device)
    model_output2 = predict_from_signal(signal, model)

    assert set(model_output.keys()) == set(model_output2.keys())
    for key in model_output:
        assert np.allclose(model_output[key], model_output2[key], atol=atol)
