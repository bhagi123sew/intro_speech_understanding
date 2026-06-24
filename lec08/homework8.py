import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    frames = np.array([
        waveform[m:m+frame_length]
        for m in range(0, len(waveform)-frame_length, step)
    ])
    return frames


def frames_to_mstft(frames):
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft


def mstft_to_spectrogram(mstft):
    spectrogram = 20 * np.log10(
        np.maximum(0.001 * np.amax(mstft), mstft)
    )
    return spectrogram


