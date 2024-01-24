import torch
import torch.nn as nn
import numpy as np



def truncated_svd(x: np.ndarray):
    """
    :param x: [N, D], matrix N > D
    :return: u: [N, rank], s: [rank], vh: [rank, D]
    """
    u, s, vh = np.linalg.svd(x.transpose() @ x)  # u.shape = D x D, s.shape = D, vh.shape = D x D
    s = np.sqrt(s)  # singular values of x
    u_times_sigma = x @ vh.transpose() 
    rank = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:rank]
    vh = vh[:rank]
    u = u_times_sigma[:, :rank] / s.reshape(1, -1)
    return u, s, vh


def logME(f: np.ndarray, y: np.ndarray):
        """
        initially implemented by You in https://github.com/thuml/LogME
        :param f: [N, D], feature matrix from pre-trained model.
        :param y: [N], target labels with element in [0, C_t).
        :return: LogME score 
        """

        f = f.astype(np.float64)
        y = y.astype(np.float64)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        N, D = f.shape
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x rank
        # s.shape = rank
        # vh.shape = rank x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)
        sigma_full_size = sigma
        if N < D:  # padding sigma to size [D]
            sigma_full_size = np.pad(sigma, ((0, D - N), (0, 0)), 'constant')

        evidences = []
        num_dim = int(y.max() + 1)
        for i in range(num_dim):
            y_ = (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
        return np.mean(evidences)


def ALogME(f: np.ndarray, y: np.ndarray):
    """
    approximated LogME 
    :param f: [N, D], feature matrix from pre-trained model.
    :param y: [N], target labels with element in [0, C_t).
    """
    f = f.astype(np.float64)
    y = y.astype(np.float64)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    N, D = f.shape
    num_dim = int(y.max() + 1)
    if N > D:  # direct SVD may be expensive
        u, s, vh = truncated_svd(f)
    else:  # N <= D
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    # u.shape = N x k
    # s.shape = k
    # vh.shape = k x D
    s = s.reshape(-1, 1)
    sigma = (s ** 2)
    sigma_full_size = sigma
    if N < D:  # padding sigma to size [D]
            sigma_full_size = np.pad(sigma, ((0, D - N), (0, 0)), 'constant')

    evidences = []
    for i in range(num_dim):

        y_ = (y == i).astype(np.float64)
        y_ = y_.reshape(-1, 1)
        z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
        z2 = z ** 2
        delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly


        t =  (sigma[0] / N)
        m2 = (sigma * z2 / ((t + sigma) ** 2)).sum()
        res2 = (z2 / ((1 + sigma / t) ** 2)).sum() + delta
        beta = N / (res2 + t * m2)
        alpha = t * beta
        evidence = D / 2.0 * np.log(alpha) \
                 + N / 2.0 * np.log(beta) \
                 - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                 - beta / 2.0 * res2 \
                 - alpha / 2.0 * m2 \
                 - N / 2.0 * np.log(2 * np.pi)
        evidences.append(evidence / N)

    return np.mean(evidences)


def assess(model, data_loader, mode='None'):
    # forward propagation and concatenate all outputs
    feature_maps = []
    labels = []
    for input, label in data_loader:
        input = input.cuda()
        feature_map = model.get_feature_map(input).mean(dim=1).cpu().detach().numpy()
        feature_maps.append(feature_map)
        label = label.cpu().detach().numpy()
        labels.append(label)
    feature_maps = np.concatenate(feature_maps, axis=0)
    labels = np.concatenate(labels, axis=0)
    # calculate model evidence
    if mode == 'LogME':
        score = logME(feature_maps, labels)
    elif mode == 'ALogME':
        score = ALogME(feature_maps, labels)
    else:
        raise NotImplementedError
    return score


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
        scores_ALogME.append(ALogME(y, labels))

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
