import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)

    energies = []
    starts = []

    for m in range(0, len(waveform) - frame_length, step):
        frame = waveform[m:m+frame_length]
        energies.append(np.sum(frame**2))
        starts.append(m)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    active = False

    for i, e in enumerate(energies):
        if e > threshold and not active:
            start = starts[i]
            active = True

        elif e <= threshold and active:
            end = starts[i] + frame_length
            segments.append(waveform[start:end])
            active = False

    if active:
        segments.append(waveform[start:])

    return segments


def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment.
    '''
    models = []

    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)

    for segment in segments:

        emphasized = np.append(
            segment[0],
            segment[1:] - 0.95 * segment[:-1]
        )

        frames = np.array([
            emphasized[m:m+frame_length]
            for m in range(0, len(emphasized)-frame_length, step)
        ])

        mstft = np.abs(np.fft.fft(frames, axis=1))

        spectrogram = 20 * np.log10(
            np.maximum(0.001 * np.amax(mstft), mstft)
        )

        lowfreq = spectrogram[:, :frame_length//2]

        model = np.mean(lowfreq, axis=0)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    sims = np.zeros((len(models), len(test_models)))

    for i in range(len(models)):
        for j in range(len(test_models)):
            sims[i, j] = np.dot(models[i], test_models[j]) / (
                np.linalg.norm(models[i]) *
                np.linalg.norm(test_models[j])
            )

    test_outputs = []

    for j in range(sims.shape[1]):
        best = np.argmax(sims[:, j])
        test_outputs.append(labels[best])

    return sims, test_outputs