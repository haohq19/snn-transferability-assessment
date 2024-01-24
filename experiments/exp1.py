import numpy as np
from ..functions.logME import logME
from ..functions.ApproxME import ApproxME


if __name__ == '__main__':
    num_samples = 100  # N
    num_dim = 10  # D
    num_classes = 10  # C_t
    # feature maps can be well classified with a linear transformation
    f = np.zeros((num_samples, num_dim))  # N x D
    for i in range(num_samples):
        for j in range(num_dim):
            if i % num_classes == j % num_classes:
                f[i][j] = np.sqrt(num_dim/num_classes)  # the Power of feature is 1
    # labels
    labels = np.arange(num_samples) % num_classes  # N
    # Gaussian noise
    noise = np.random.normal(0, 1, (num_samples, num_dim))  # N x D, std = 1, the Power of Gaussian noise is var = 1
    # add f and noise with 100 different SNRs
    snrs_db = np.linspace(-10, 10, 100)  # SNR in dB
    snrs = 10 ** (np.array(snrs_db) / 10)  # SNR
    scores_logME = []
    scores_ALogME = []
    for snr in snrs:
        y = f + noise / snr
        # normalize each y
        for i in range(num_samples):
            y[i] = y[i] / np.linalg.norm(y[i])
        scores_logME.append(logME(y, labels))
        scores_ALogME.append(ApproxME(y, labels))

    # draw figure
    import matplotlib.pyplot as plt
    # change the font to Arial
     
    plt.figure()
    # black line
    plt.plot(snrs_db, scores_logME, label='MacKay\'s Algorithm', c='black', ls='-')
    # red solid line
    plt.plot(snrs_db, scores_ALogME, label='Ours', c='red', ls='-.')
    plt.xlabel('SNR (dB)')
    plt.ylabel('score')
    plt.legend()
    # text
    plt.text(8, - 0.5, 'N = {}'.format(num_samples))
    plt.text(8, - 0.7, 'D = {}'.format(num_dim))
    # save figure
    plt.savefig('MacKay_Ours_Comparison_N{}_D{}.png'.format(num_samples, num_dim))
