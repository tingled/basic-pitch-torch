from basic_pitch_torch.inference import predict
from basic_pitch_torch.constants import AUDIO_SAMPLE_RATE
from basic_pitch_torch.model import BasicPitchTorch
from basic_pitch_torch import note_creation as infer

import librosa
from matplotlib import pyplot as plt
import torch
import scipy

sr = 22050

audio_path = librosa.ex("fishin")

# orig
model_output, midi_data, note_events = predict(audio_path)

out = midi_data.synthesize(fs=sr)
scipy.io.wavfile.write("orig.wav", sr, out)
orig_piano_roll = midi_data.get_piano_roll()  # (128, N), with N sampled at 100hz

start_sec = 10
end_sec = 15
notes = model_output["note"][86 * start_sec : 86 * end_sec].T
notes[notes < 0.5] = 0
piano = orig_piano_roll[:, 100 * start_sec : 100 * end_sec]

# plot piano roll and notes
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.imshow(piano, aspect="auto", origin="lower")
plt.subplot(2, 1, 2)
plt.imshow(notes, aspect="auto", origin="lower")
plt.show()
