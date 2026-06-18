import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.
    '''
    G = np.power(2, 1/12)

    n = np.arange(Fs // 2)

    root = np.cos(2 * np.pi * f * n / Fs)
    major_third = np.cos(2 * np.pi * (f * G**4) * n / Fs)
    major_fifth = np.cos(2 * np.pi * (f * G**7) * n / Fs)

    x = root + major_third + major_fifth

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    '''
    W = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            W[k, n] = np.cos(2*np.pi*k*n/N) - 1j*np.sin(2*np.pi*k*n/N)

    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    X = np.abs(np.fft.fft(x))

    freqs = []

    for i in range(3):
        k = np.argmax(X)
        f = k * Fs / len(x)
        freqs.append(f)
        X[k] = 0

    freqs.sort()

    return freqs[0], freqs[1], freqs[2]