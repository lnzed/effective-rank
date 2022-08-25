import os
import unittest
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

import logging


class EffectiveRankCalculator():

    def __init__(self,
                algorithm=None,
                minRank=1,
                tickers=[],
                period=252*2,
                pctChangePeriod=21,
                covar_method='ledoit-wolf',
                resolution=None) -> None:

        self.algorithm = algorithm
        self.minRank = minRank
        self.tickers = tickers
        self.period = period
        self.pctChangePeriod = pctChangePeriod
        self.resolution = resolution
        self.covar_method = covar_method

        self.effectiveRank = 0
        self.effectiveRankRaw = 0
        self.effectiveRankCeil = 0
        self.erank_history = None
        self.U = None
        self.D = None
        self.VT = None

        self.logger = logging.getLogger(__name__)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)
        return

    def calculate(self, covarianceMatrix=None, prices=None):
        
        if covarianceMatrix is None and prices is None and self.algorithm is None:
            self.logger.error("Cannot calculate effective rank")
            return self

        if covarianceMatrix is None:
            covarianceMatrix = self.getCovarianceMatrix(prices=prices)

        self.U, self.D, self.VT, self.effectiveRankRaw = effectiveRank(covarianceMatrix)
        self.effectiveRank = max(self.minRank, self.effectiveRankRaw)
        self.effectiveRankCeil = int(np.ceil(self.effectiveRank))

        return self

    def calculate_history(self, covariance_history=None, prices=None):
        if covariance_history is None and prices is None:
            self.logger.error("")
            return self
        if covariance_history is None:
            covariance_history = prices.dropna().pct_change(self.pctChangePeriod).dropna().rolling(self.period).cov().dropna()#apply(self.getCovarianceMatrix)
        cov_mats = [covMat.values for k, covMat in covariance_history.groupby("Date")]
        self.erank_history = pd.Series([effectiveRank(covMat)[-1] for covMat in cov_mats], index=sorted(set(covariance_history.reset_index().Date), reverse=False))
        return self

    def getCovarianceMatrix(self, prices=None):

        if self.algorithm is None and prices is None:
            self.logger.error("No data available to calculate the covariance matrix. Cannot calculate the effective rank. ")
            return None

        if prices is None:
            prices = self.algorithm.History(self.tickers, self.period, self.resolution)
            prices = prices['close'].unstack(level=0)

        prices = prices.dropna().pct_change(self.pctChangePeriod).dropna()

        if (len(prices) == 0):
            self.logger.error("No data left after calculating percent changes. Cannot calculate the effective rank. ")

        if self.covar_method == "ledoit-wolf":
            covMatrix = ledoit_wolf(prices)[0]
        else:
            covMatrix = prices.cov().dropna()

        return covMatrix


def svd(A):
    return np.linalg.svd(A)  # returns U, D, VT


def shannonEntropy(D):
    entropy = 0
    _sum = D.sum()
    if _sum == 0 or np.isnan(_sum):
        return 0
    if np.NaN in D:
        return 0
    pD = D / _sum
    try:
        entropy = np.exp(-np.sum(pD*np.log(pD)))
    except Exception as e:
        print(e)
        pass
    return entropy


def effectiveRank(matrix):
    U, D, VT = svd(matrix)
    rank = shannonEntropy(D)
    return U, D, VT, rank


class TestCases(unittest.TestCase):

    def setUp(self):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effective_rank_test_data.csv")
        self.prices = pd.read_csv(filepath).set_index("Date")
        self.A = np.array([[3, 4, 3], [1, 2, 3], [4, 2, 1]])
        self.effectiveRankCalculator = EffectiveRankCalculator()
        return

    def testSVD(self):
        U, D, VT = svd(self.A)
        print("U:")
        print(U)
        print("D:")
        print(D)
        print("VT:")
        print(VT)
        return

    def testShannonEntropy(self):
        U, D, VT = svd(self.A)
        print("Shannon entropy:")
        print(shannonEntropy(D))
        return

    def testEffectiveRankCalculate(self):
        self.effectiveRankCalculator.calculate(covarianceMatrix=self.A)
        self.effectiveRankCalculator.calculate(prices=self.prices)
        print(self.effectiveRankCalculator.__dict__)


if __name__ == '__main__':
    unittest.main()
