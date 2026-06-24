import numpy as np
import torch
import torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    '''

    L = int(0.004 * Fs)
    S = int(0.002 * Fs)

    mstft = np.abs(
        np.fft.fft(
            np.array([
                waveform[m+1:m+1+L] - waveform[m:m+L]
                for m in range(0, len(waveform)-L, S)
            ]),
            axis=1
        )
    )

    features = 20 * np.log10(
        np.maximum(0.001 * np.amax(mstft), mstft)
    )[:, 0:int(L/2)]

    framelength = int(0.025 * Fs)
    frameskip = int(0.01 * Fs)

    energy = np.sum(
        np.square(
            np.array([
                waveform[m:m+framelength]
                for m in range(0, len(waveform)-framelength, frameskip)
            ])
        ),
        axis=1
    )

    VAD = [1 if energy[m] > 0.1 * max(energy) else 0
           for m in range(len(energy))]

    startframes = [
        m for m in range(1, len(VAD))
        if VAD[m-1] == 0 and VAD[m] == 1
    ]

    endframes = [
        m for m in range(1, len(VAD))
        if VAD[m-1] == 1 and VAD[m] == 0
    ]

    labels = np.zeros(len(features))

    for num in range(1, 6):
        labels[5*startframes[num-1]:5*endframes[num-1]+4] = num

    return features, labels.astype(int)


def train_neuralnet(features, labels, iterations):
    '''
    Train neural network.
    '''

    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = torch.nn.Sequential(
        torch.nn.LayerNorm(nfeats, dtype=float),
        torch.nn.Linear(nfeats, nlabels, dtype=float)
    )

    x = torch.tensor(features, dtype=float)
    y = torch.tensor(labels, dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Test neural network.
    '''

    x = torch.tensor(features, dtype=float)

    with torch.no_grad():
        outputs = model(x)
        probabilities = outputs.softmax(-1).detach().numpy()

    return probabilities