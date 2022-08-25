# Effective Rank

Calculate the effective rank of a covariance matrix (see https://infoscience.epfl.ch/record/110188/files/RoyV07.pdf)

The main class (EffectiveRank.py) implements the calculation of the effective rank of a covariance matrix. The calculation can be performed either by directly providing a covariance matrix, or by providing a Pandas DataFrame.
In the latter case, each column in the dataframe shall contain a timeseries, for example:

```
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
```

The calculation can be made:
1. for a single point in time using the entire history provided, or
2. in a rolling fashion to obtain efficient_rank = f(time)

Example usage:

```
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
    
```
