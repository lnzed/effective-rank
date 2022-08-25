import os
import unittest
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

import logging


class EffectiveRankCalculator():
    """ Calculate the effective rank of a covariance matrix (see https://infoscience.epfl.ch/record/110188/files/RoyV07.pdf)

    This class implements the calculation of the effective rank of a covariance matrix.
    The calculation can be performed either by directly providing a covariance matrix,
    or by providing a Pandas DataFrame.
    In the latter case, each column in the dataframe shall contain a timeseries, for example:

                XLB        XLE        XLF         XLI         XLK  
    Date                                                                  
    1998-12-22  20.828125  23.265625  18.937855   23.281250   32.046875   
    1998-12-23  21.046875  23.750000  19.217100   23.687500   32.812500   
    1998-12-24  21.531250  23.625000  19.344028   24.000000   32.687500   
    1998-12-28  21.343750  23.500000  19.090172   24.125000   32.781250   
    1998-12-29  21.734375  23.734375  19.293259   24.468750   32.875000   
    ...               ...        ...        ...         ...         ...   
    2022-08-18  80.650002  79.480003  35.669998  100.129997  150.850006   
    2022-08-19  79.169998  79.459999  34.959999   98.839996  148.110001   
    2022-08-22  77.910004  79.309998  34.189999   96.959999  143.979996   
    2022-08-23  78.680000  82.169998  34.049999   97.139999  143.600006   
    2022-08-24  78.849998  83.180000  34.230000   97.459999  143.690002

    The calculation can be made:
    1. for a single point in time using the entire history provided, or
    2. in a rolling fashion to obtain efficient_rank = f(time)

    Example usage:
    
    import importlib
    import numpy as np
    import pandas as pd
    import yfinance as yf

    from effectiverank import EffectiveRankCalculator

    data = yf.download(tickers = "XLK,XLV,XLP,XLY,XLI,XLU,XLB,XLE,XLF".split(","), period = 'max')
    calculator = EffectiveRankCalculator.EffectiveRankCalculator()
    calculator.calculate(prices=data['Close'])
    calculator.effectiveRank
    >> 3.4527689261842256

    calculator.calculate_history(prices=data['Close'])
    calculator.erank_history
    >>
        2001-01-22    5.150895
        2001-01-23    5.142469
        2001-01-24    5.134840
        2001-01-25    5.135085
        2001-01-26    5.135292
                        ...   
        2022-08-18    3.695294
        2022-08-19    3.696491
        2022-08-22    3.696219
        2022-08-23    3.694735
        2022-08-24    3.692226
        Length: 5433, dtype: float64

    """

    def __init__(self,
                minRank=1,
                period=252*2,
                pctChangePeriod=21,
                covar_method='ledoit-wolf',
                algorithm=None,
                tickers=[],
                resolution=None
                ) -> None:
        """Constructor

        Args:
            minRank (int, optional): Minimum rank. Defaults to 1.
            period (_type_, optional): Lookback period to calculate the covariance matrix. Defaults to 252*2.
            pctChangePeriod (int, optional): Period for calculating the percent change. Passed to pd.DataFrame().pct_change(period=*pctChangePeriod*).
                Defaults to 21.
            covar_method (str, optional): Covariance calculation method. Defaults to 'ledoit-wolf' (see scikit-learn).
            algorithm (QCAlgorithm, optional): A QCAlgorithm instance. Defaults to None.
                If provided, the timeseries for a given security will be automatically fetched
            tickers (list, optional): Tickers representing the securities to fetch through the QCAlgorithm. Defaults to [].
                Example: ['QQQ','TLT','SPY']
            resolution (_type_, optional): Resolution of the data to fetch through QCAlgorithm. Defaults to None.
        """

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
