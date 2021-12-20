#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import linalg, fftpack
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class ChannelSelection():
    def BIC(self, X, p_max=20):
        p, bic = self.compute_order(X, p_max=p_max)
        return p

    def compute_order(self, X, p_max):
        """Estimate AR order with BIC
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        p_max : int
            The maximum model order to test
        Returns
        -------
        p : int
            Estimated order
        bic : ndarray, shape (p_max + 1,)
            The BIC for the orders from 0 to p_max.
        """
        N, n = X.shape

        bic = np.empty(p_max + 1)
        bic[0] = np.inf # XXX

        Y = X.T

        for p in range(1, p_max + 1):
            # print(p)
            A, sigma = self.mvar_fit(X, p)
            A_2d = np.concatenate(A, axis=1)

            n_samples = n - p
            bic[p] = n_samples * N * math.log(2. * math.pi)
            bic[p] += n_samples * np.log(linalg.det(sigma))
            bic[p] += p * (N ** 2) * math.log(n_samples)

            sigma_inv = linalg.inv(sigma)
            S = 0.
            for i in range(p, n):
                res = Y[i] - np.dot(A_2d, Y[i - p:i][::-1, :].ravel())
                S += np.dot(res, sigma_inv.dot(res))

            bic[p] += S

        max = np.inf
        p = 0
        for i in range(2, p_max + 1):
            if not np.isnan(bic[i]) and bic[i] < max:
                max = bic[i]
                p = i   
        # p = np.argmin(bic)
        return p, bic
    
    def cov(self, X, p):
        """vector autocovariance up to order p
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        Returns
        -------
        R : ndarray, shape (p + 1, N, N)
            The autocovariance up to order p
        """
        N, n = X.shape
        R = np.zeros((p + 1, N, N))
        for k in range(p + 1):
            R[k] = (1. / float(n - k)) * np.dot(X[:, :n - k], X[:, k:].T)
        return R

    def mvar_fit(self, X, p):
        """Fit MVAR model of order p using Yule Walker
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        n_fft : int
            The length of the FFT
        Returns
        -------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        sigma : array, shape (N,)
            The noise for each time series
        """
        N, n = X.shape
        gamma = self.cov(X, p)  # gamma(r,i,j) cov between X_i(0) et X_j(r)
        G = np.zeros((p * N, p * N))
        gamma2 = np.concatenate(gamma, axis=0)
        gamma2[:N, :N] /= 2.

        for i in range(p):
            G[N * i:, N * i:N * (i + 1)] = gamma2[:N * (p - i)]

        G = G + G.T  # big block matrix

        gamma4 = np.concatenate(gamma[1:], axis=0)

        phi = linalg.solve(G, gamma4)  # solve Yule Walker

        tmp = np.dot(gamma4[:N * p].T, phi)
        sigma = gamma[0] - tmp - tmp.T + np.dot(phi.T, np.dot(G, phi))

        phi = np.reshape(phi, (p, N, N))
        for k in range(p):
            phi[k] = phi[k].T

        return phi, sigma

    def spectral_density(self, A, n_fft=None):
        """Estimate PSD from AR coefficients
        Parameters
        ----------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        n_fft : int
            The length of the FFT
        Returns
        -------
        fA : ndarray, shape (n_fft, N, N)
            The estimated spectral density.
        """
        p, N, N = A.shape
        if n_fft is None:
            n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
        A2 = np.zeros((n_fft, N, N))
        A2[1:p + 1, :, :] = A  # start at 1 !
        fA = fftpack.fft(A2, axis=0)
        freqs = fftpack.fftfreq(n_fft)
        I = np.eye(N)

        for i in range(n_fft):
            fA[i] = linalg.inv(I - fA[i])

        return fA, freqs
    
    def PDC(self, A, sigma=None, n_fft=None):
        """Partial directed coherence (PDC)
        Parameters
        ----------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        sigma : array, shape (N,)
            The noise for each time series.
        n_fft : int
            The length of the FFT.
        Returns
        -------
        P : ndarray, shape (n_fft, N, N)
            The estimated PDC.
        """
        p, N, N = A.shape

        if n_fft is None:
            n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

        H, freqs = self.spectral_density(A, n_fft)
        P = np.zeros((n_fft, N, N))

        if sigma is None:
            sigma = np.ones(N)

        for i in range(n_fft):
            B = H[i]
            B = linalg.inv(B)
            V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
            V = np.diag(V)  # denominator squared
            P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]

        return P, freqs

    def select(self, X, select_num=10):
        p = self.BIC(X)
        A_est, sigma = self.mvar_fit(X, p)
        sigma = np.diag(sigma)
        P, freqs = self.PDC(A_est, sigma)
        m, N, N = P.shape
        pdc = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                pdc[i, j] = np.mean(P[freqs >= 0, i, j])
        pdc_avg = []
        pdc_sum = pdc.sum(axis=0)
        for i in range(N):
            pdc_avg.append((pdc_sum[i]-pdc[i,i])/(N-1))
        channel_selected = np.argpartition(pdc_avg, -select_num)[-select_num:]
        return channel_selected