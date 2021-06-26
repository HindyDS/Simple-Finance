#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader as rd
from scipy.stats import t
from sklearn.mixture import GaussianMixture
from scipy.optimize import linprog
from scipy.optimize import minimize
import plotly.graph_objects as go
from scipy.stats import probplot
from scipy.stats import jarque_bera, normaltest
from scipy.stats import t
from scipy.stats import kstest
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats import kurtosis
import pmdarima as pm
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta
today = date.today()
from hmmlearn import hmm
import yfinance as yf

class finance:
    def __init__(self, ticks, start, end):
        adj = pd.DataFrame()
        stocks_list=[]
        for tick in ticks:
            df = yf.download(tick, start=start, end=end)
            df['sym'] = tick
            df['log_ret'] = np.log1p(df[df['sym'] == tick].loc[:,'Adj Close'].pct_change(1))
            df.reset_index(inplace=True)
            df.dropna(axis=0, how='all', inplace=True) 
            
            stocks_list.append(df)
            stocks = pd.concat(stocks_list, axis=0)
                    
        self.stocks = stocks
        
        '''drop_list = []
        for i in self.stocks['sym'].unique():
            if len(self.stocks[self.stocks['sym']==i]) < 2770/2:
                drop_list.append(i)
                print(f'{i} has been dropped due to its short existence.')'''

        self.stocks.set_index('sym', inplace=True)
        '''self.stocks.drop(drop_list, axis=0, inplace=True)'''
        self.stocks.reset_index(inplace=True)
        
        self.returns = pd.pivot_table(self.stocks, 
                                      values='log_ret', 
                                      index='Date', 
                                      columns=['sym'])
        
        self.ticks = self.stocks['sym'].unique()
    
    def stock(self, sym):
        return self.stocks[self.stocks['sym'] == sym]
        
    def stats(self, window=None):              
        stats = pd.DataFrame()
        window = window
        if window == None:
            window=len(self.stocks)
        for i,j in zip(self.stocks['sym'].unique(), range(len(self.stocks['sym'].unique()))):
            stats.loc[j,'sym'] = i
            stats.loc[j,f'Age'] = len(self.stocks )- self.stocks['Close'].isnull().sum()
            stats.loc[j,f'Mean of past {window} days'] = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).mean()
            stats.loc[j,f'Variance of past {window} days'] = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).var()
            stats.loc[j,f'Standard Deviation of past {window} days'] = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).std(ddof=1)
            stats.loc[j,f'kurtosis of past {window} days'] = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).kurtosis()
            stats.loc[j,f'Skewness of past {window} days'] = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).skew()
            values = self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna().to_numpy()
            def cdf(x):
                return t.cdf(x,  t.fit(values)[0], t.fit(values)[1], t.fit(values)[2])

            stats.loc[j,f'p of Kolmogorov–Smirnov test of past {window} days'] = round(kstest(self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna(), cdf)[1], 4)
            stats.loc[j,f'p of Jarque-Bera of past {window} days'] = round(jarque_bera(self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna())[1], 4)
            stats.loc[j,f'p of ttest_1samp_p of past {window} days'] = round(ttest_1samp(self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna(), 0)[1], 4)
            stats.loc[j,f'p of Normal test of past {window} days'] = round(normaltest(self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna())[1], 4)
            stats.loc[j,f'p of Adfuller test of past {window} days'] = round(ts.adfuller(self.stocks[self.stocks['sym'] == i].loc[:, 'log_ret'].tail(window).dropna())[1], 4)
                                      
            stats.loc[j, f'p of Kolmogorov–Smirnov test of past {window} days'] = np.where(stats[stats['sym']==i][f'p of Jarque-Bera of past {window} days'] >= 0.5, 't-test is not a decent fit', 't-test is a decent fit')
            stats.loc[j, f'p of Jarque-Bera of past {window} days'] = np.where(stats[stats['sym']==i][f'p of Jarque-Bera of past {window} days'] >= 0.5, 'Normally distributed', 'Not Normally distributed')
            stats.loc[j, f'p of ttest_1samp_p of past {window} days'] = np.where(stats[stats['sym']==i][f'p of ttest_1samp_p of past {window} days'] >= 0.5, 'Random sample is not equal to true mean', 'Random sample is equal to true mean')
            stats.loc[j, f'p of Normal test of past {window} days'] = np.where(stats[stats['sym']==i][f'p of Normal test of past {window} days'] >= 0.5, 'Normally distributed', 'Not Normally distributed')
            stats.loc[j, f'p of Adfuller test of past {window} days'] = np.where(stats[stats['sym']==i][f'p of Adfuller test of past {window} days'] >= 0.5, 'Not Stationary', 'Stationary')

            if window < 2000:
                stats.drop(f'p of Jarque-Bera of past {window} days', axis=1, inplace=True)
            
        stats.to_csv('stats.csv',index = False)
        return stats
    
    def info(self):
        useless_list = ['state', 'maxAge', 'shortName', 'longName', 'logo_url', 'volume24Hr', 'navPrice', 'totalAssets', 
                'toCurrency','expireDate','yield','algorithm','circulatingSupply', 'startDate', 'lastMarket',
                'maxSupply','openInterest','volumeAllCurrencies','strikePrice','ytdReturn', 'fromCurrency', 
                'annualHoldingsTurnover','beta3Year','morningStarRiskRating', 'revenueQuarterlyGrowth',
                'fundInceptionDate','annualReportExpenseRatio','fundFamily','threeYearAverageReturn',
                'legalType','morningStarOverallRating','lastCapGain','impliedSharesOutstanding', 
                'category','fiveYearAverageReturn', 'lastSplitDate', 'lastSplitFactor', 'lastDividendDate',
                'underlyingSymbol','underlyingExchangeSymbol','headSymbol', 'address2', 'uuid',
               'companyOfficers', 'zip', 'longBusinessSummary', 'city', 'phone', 'website', 'fax', 'dividendRate' , 
                'dividendYield', 'address1']

        syms = self.ticks
        res = []
        for sym in syms:
            info = yf.Ticker(sym).info
            res.append(pd.DataFrame([info]))
            df = pd.concat(res)

        for i in useless_list:
            try:
                df.drop(i, axis=1, inplace=True)
            except KeyError:
                continue
                
        df.index = range(0, len(df))
        # The tighter or narrower the spread, the more liquid the transactions. Conversely, a larger spread is more illiquid.
        df['BidAskSpread'] = (df['bid'] - df['ask'])/df['ask'] * 100
        
        screen = ['quoteType', 'symbol', 'sector', 'country', 'industry',
          'trailingAnnualDividendYield', 'trailingAnnualDividendRate', 'lastDividendValue', 'payoutRatio',  
          'trailingPE', 'forwardPE', 'pegRatio', 
          'profitMargins', 'forwardEps', 'trailingEps', 'enterpriseToRevenue', 'enterpriseToEbitda', 'mostRecentQuarter',
          'priceToSalesTrailing12Months', 
          'marketCap', 'bookValue', 'enterpriseValue',
          'sharesOutstanding', 'floatShares',  'shortRatio', 'shortPercentOfFloat', 'beta', 'earningsQuarterlyGrowth', 
          'sharesPercentSharesOut', 'heldPercentInstitutions', 'netIncomeToCommon', 'priceToBook', 'heldPercentInsiders'] 
        
        other = ['fullTimeEmployees',  'previousClose', 'regularMarketOpen',
                 'twoHundredDayAverage', 'regularMarketDayHigh', 'averageDailyVolume10Day', 'regularMarketPreviousClose',
                 'fiftyDayAverage', 'open', 'averageVolume10days', 'exDividendDate', 'regularMarketDayLow', 'priceHint',
                 'currency', 'regularMarketVolume', 'averageVolume', 'dayLow', 'ask', 'askSize', 'volume', 'fiftyTwoWeekHigh',
                 'fiveYearAvgDividendYield', 'fiftyTwoWeekLow', 'bid', 'tradeable', 'bidSize', 'dayHigh', 'exchange',
                 'exchangeTimezoneName', 'exchangeTimezoneShortName', 'isEsgPopulated', 'gmtOffSetMilliseconds', 'messageBoardId',
                 'market', '52WeekChange', 'sharesShort', 'lastFiscalYearEnd', 'SandP52WeekChange', 'nextFiscalYearEnd',
                 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesShortPriorMonth', 'regularMarketPrice']
        
        return df[screen], df[other]
    
    def port_opt(self, sym=None, plot=True):
        if sym == None:
            sym = self.returns.columns
        mean_return = self.returns[sym].mean()

        cov = self.returns[sym].cov()
        cov_np = cov.to_numpy()
        N = 10000
        D = len(mean_return)
        returns = np.zeros(N)
        risks = np.zeros(N)
        random_weights = []
        for i in range(N):
            rand_range = 1.0
            w = np.random.random(D)*rand_range - rand_range / 2 # with short-selling
            w[-1] = 1 - w[:-1].sum()
            np.random.shuffle(w)
            random_weights.append(w)
            ret = mean_return.dot(w)
            risk = np.sqrt(w.dot(cov_np).dot(w))
            returns[i] = ret
            risks[i] = risk
            
        single_asset_returns = np.zeros(D)
        single_asset_risks = np.zeros(D)
        for i in range(D):
            ret = mean_return[i]
            risk = np.sqrt(cov_np[i,i])

            single_asset_returns[i] = ret
            single_asset_risks[i] = risk

        D = len(mean_return)
        A_eq = np.ones((1,D))
        b_eq = np.ones(1)
        ### Note: The bounds are by default (0, None) unless otherwise specified.
        # bounds = None
        bounds = [(-0.5, None)]*D
        # minimize
        res = linprog(mean_return, A_eq=A_eq,  b_eq=b_eq, bounds=bounds)
        min_return = res.fun
        # maximize
        res = linprog(-mean_return, A_eq=A_eq,  b_eq=b_eq, bounds=bounds)
        max_return = -res.fun

        N = 100
        target_returns = np.linspace(min_return, max_return, num=N)

        def get_portfolio_variance(weights):
            return weights.dot(cov).dot(weights)

        def target_return_constraint(weights, target):
            return weights.dot(mean_return) - target

        def portfolio_constraint(weights):
            return weights.sum() - 1

        constraints = [
            {
                'type': 'eq',
                'fun': target_return_constraint,
                'args': [target_returns[0]], # will be updated in loop
            },
            {
                'type': 'eq',
                'fun': portfolio_constraint,
            }
        ]

        # check if it works
        res = minimize(fun=get_portfolio_variance,
                      x0=np.ones(D) / D, # uniform, start point array, initial guess for the weights
                      method='SLSQP',
                      constraints=constraints,
                      )

        optimized_risks = []
        for target in target_returns: # target_returns = np.linspace(min_return, max_return, num=N)
            # set target return constraint
            constraints[0]['args'] = [target] # 'args': [target_returns[0]], update constraints from target_returns

            res = minimize(
                      fun=get_portfolio_variance,
                      x0=np.ones(D) / D, # uniform, start point array, initial guess for the weights
                      method='SLSQP',
                      constraints=constraints,
                      bounds=bounds
            )
            optimized_risks.append(np.sqrt(res.fun))
            if res.status != 0:
                print(res)

        # Min variance portfolio
        # Let's limit the magnitude of the weights
        res = minimize(
                      fun=get_portfolio_variance,
                      x0=np.ones(D) / D, # uniform, start point array, initial guess for the weights
                      method='SLSQP',
                      constraints={
                          'type': 'eq',
                          'fun': portfolio_constraint,
                      },
                      bounds=bounds
            )

        mv_risk = np.sqrt(res.fun)
        mv_weights =res.x
        mv_ret = mv_weights.dot(mean_return)
        
        risk_free_rate = 0.03/252

        def neg_sharpe_ratio(weights):
            mean = weights.dot(mean_return)
            sd = np.sqrt(weights.dot(cov).dot(weights))
            return -(mean - risk_free_rate) / sd

        res = minimize(fun=neg_sharpe_ratio,
                      x0=np.ones(D) / D, # uniform
                      method='SLSQP',
                      constraints={
                          'type':'eq',
                          'fun': portfolio_constraint,
                      },
                      bounds=bounds,
                      )

        best_sr, best_w = -res.fun, res.x

        mc_best_w = None
        mc_best_sr = float('-inf')
        for i, (risk, ret) in enumerate(zip(risks, returns)):
            sr = (ret - risk_free_rate) / risk
            if sr > mc_best_sr:
                mc_best_sr = sr
                mc_best_w = random_weights[i]
                
        if plot == True:
            fig, ax = plt.subplots(figsize=(10, 5))
            # found by optimization
            opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
            opt_ret = mean_return.dot(best_w)
            plt.scatter([opt_risk], [opt_ret], c='green'); 
        
        
            # tangent line
            x1 = 0
            y1 = risk_free_rate
            x2 = opt_risk
            y2 = opt_ret
            plt.plot([x1, x2], [y1, y2], color='orange')
            plt.plot([x2, x2 * 3], [y2, (y2 - y1) / (x2 - x1) * x2 * 3], color='orange', linestyle='dashed')

            plt.scatter(risks, returns, alpha=0.1);
            plt.scatter(single_asset_risks, single_asset_returns, c='red'); # Single Asset
            plt.plot(optimized_risks, target_returns, c='black'); # Efficient Frontier
            plt.scatter([mv_risk], [mv_ret], c='purple') # Global Minimum of Risk
            plt.grid(False)
            plt.xlabel('Risks (Covariance)')
            plt.ylabel('Returns (Mean)')
            plt.legend(['Tangency Portoflio', 
                        'Borrow Cash at the Risk-free rate', 
                        'Efficient Frontier', 
                        'Risk-free asset', 
                        'Portoflios', 
                        'Single Asset', 
                        'Global Minimum of Risk'],
                       bbox_to_anchor=(1.05, 1), loc='upper left')

            sns.despine()

            print(f'Global Minimum of Risk: {round(mv_risk, 4)}')
            print(f'Return of Global Minimum of Risk: {round(mv_ret, 4)}')
            print(' ')
            print('Tick and Weights: ')
            for tick, weight in zip(self.ticks, mc_best_w):
                print(tick, f' ({round(weight*100, 4)}%)')
            print(' ')
            print(f'Sharpe Raito: {round(mc_best_sr, 4)}')
        
        if plot == False:
            return round(mc_best_sr, 4)
        
    def plot_candles(self, sym): 
        fig = go.Figure(data=[
            go.Candlestick(x=self.stocks[self.stocks['sym'] == sym]['Date'],
            open=self.stocks[self.stocks['sym'] == sym]['Open'],
            high=self.stocks[self.stocks['sym'] == sym]['High'],
            low=self.stocks[self.stocks['sym'] == sym]['Low'], 
            close=self.stocks[self.stocks['sym'] == sym]['Adj Close'],
            name='Candlestick')])
        
        fig.update_layout(yaxis_title='USD ($)')
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()
        
    def sma(self, sym, window_slow=200, window_fast=50, test_size=0.8, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]

        train['SlowSMA'] = train['Close'].rolling(window_slow).mean()
        train['FastSMA'] = train['Close'].rolling(window_fast).mean()
            
        test['SlowSMA'] = test['Close'].rolling(window_slow).mean()
        test['FastSMA'] = test['Close'].rolling(window_fast).mean()    
            
        train.loc[:,'Signal'] = np.where(train.loc[:,'FastSMA'] >= train.loc[:,'SlowSMA'], 1, 0)
        test.loc[:,'Signal'] = np.where(test.loc[:,'FastSMA'] >= test.loc[:,'SlowSMA'], 1, 0)

        train.loc[:,'PrevSignal'] = train.loc[train.iloc[:-1].index,'Signal'].shift(1)
        train.loc[:,'Buy'] = (train.loc[:,'PrevSignal'] == 0) & (train.loc[:,'Signal'] == 1) # Fast < Slow --> Fast > Slow
        train.loc[:,'Sell'] = (train.loc[:,'PrevSignal'] == 1) & (train.loc[:,'Signal'] == 0) # Fast > Slow --> Fast < Slow
        
        test.loc[:,'PrevSignal'] = test.loc[test.iloc[:-1].index,'Signal'].shift(1)
        test.loc[:,'Buy'] = (test.loc[:,'PrevSignal'] == 0) & (test.loc[:,'Signal'] == 1) # Fast < Slow --> Fast > Slow
        test.loc[:,'Sell'] = (test.loc[:,'PrevSignal'] == 1) & (test.loc[:,'Signal'] == 0) # Fast > Slow --> Fast < Slow
        
        is_invested = False
        train.loc[:,'IsInvested'] = np.where(train.loc[:,'Sell'] == True, False, True)
        train.loc[:,'AlgoLogReturn'] = train.loc[:,'IsInvested'] * train.loc[:,'log_ret']
        
        is_invested = False
        test.loc[:,'IsInvested'] = np.where(test.loc[:,'Sell'] == True, False, True)
        test.loc[:,'AlgoLogReturn'] = test.loc[:,'IsInvested'] * test.loc[:,'log_ret']
        
        self.algoreturn_train_SMA = train.loc[train.index[:-1],'AlgoLogReturn'].sum()
        self.algoreturn_test_SMA = test.loc[test.index[:-1],'AlgoLogReturn'].sum()
        
        self.train_sma = train
        self.test_sma = test
        
        if plot == True:
            fig = go.Figure(data=[
                go.Candlestick(x=self.stocks[self.stocks['sym'] == sym]['Date'],
                open=self.stocks[self.stocks['sym'] == sym]['Open'],
                high=self.stocks[self.stocks['sym'] == sym]['High'],
                low=self.stocks[self.stocks['sym'] == sym]['Low'], 
                close=self.stocks[self.stocks['sym'] == sym]['Close'],
                name='Candlestick')])

            # Create and style traces
            fig.add_trace(go.Scatter(x=self.train[self.train['sym'] == sym]['Date'],
                                     y=self.train[self.train['sym'] == sym]['SlowEMA'], 
                                     name = 'SlowEMA (train)',
                                     line=dict(color='orange', width=1)))

            fig.add_trace(go.Scatter(x=self.train[self.train['sym'] == sym]['Date'], 
                                     y=self.train['FastEMA'],
                                     name='FastEMA (train)',
                                     line=dict(
                                         color='yellow', 
                                         width=1))) # dash options include 'dash', 'dot', and 'dashdot'

            fig.add_trace(go.Scatter(x=self.test[self.test['sym'] == sym]['Date'],
                                     y=self.test[self.test['sym'] == sym]['SlowEMA'], 
                                     name = 'SlowEMA (Test)',
                                     line=dict(
                                         color='orange', 
                                         width=1,
                                         dash='dash')))

            fig.add_trace(go.Scatter(x=self.test[self.test['sym'] == sym]['Date'],
                                     y=self.test[self.test['sym'] == sym]['FastEMA'],
                                     name='FastEMA (Test)',
                                     line=dict(
                                         color='yellow', 
                                         width=1,
                                         dash='dash'))) # dash options include 'dash', 'dot', and 'dashdot'


            fig.add_trace(go.Scatter(x=self.train[self.train['sym'] == sym]['Date'],
                                     y=self.train[self.train['sym'] == sym]['Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3))) 

            fig.add_trace(go.Scatter(x=self.train[self.train['sym'] == sym]['Date'],
                                     y=self.train[self.train['sym'] == sym]['Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)))  

            fig.add_trace(go.Scatter(x=self.test[self.test['sym'] == sym]['Date'],
                                     y=self.test[self.test['sym'] == sym]['Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3))) 

            fig.add_trace(go.Scatter(x=self.test[self.test['sym'] == sym]['Date'],
                                     y=self.test[self.test['sym'] == sym]['Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)))  

            # Edit the layout    
            fig.update_layout(
                xaxis_rangeslider_visible=False, 
                title='Exponential Moving Average',
                xaxis_title='Date',
                yaxis_title='USD ($)', 
                plot_bgcolor='#fafafa')
            fig.update_xaxes(showgrid=True, gridcolor='#b3b3b3')
            fig.update_yaxes(showgrid=True, gridcolor='#b3b3b3')
            fig.show()
            
            sma_res[sym] = [self.algoreturn_train_SMA, self.algoreturn_test_SMA]
            
        return self.algoreturn_train_SMA, self.algoreturn_test_SMA
    
    def sma_gs(self, sym, search_para_slow=(51, 200), search_para_fast=(1, 50)):
        sma_train_return = []
        sma_test_return = []
        slow_ = []
        fast_ = []
        for slow in range(search_para_slow[0], search_para_slow[1]):
            for fast in range(search_para_fast[0], search_para_fast[1]):
                sma_train_return.append(fin.sma(sym, window_slow=slow, window_fast=fast, test_size=0.8)[0])
                sma_test_return.append(fin.sma(sym, window_slow=slow, window_fast=fast, test_size=0.8)[1])
                slow_.append(slow)
                fast_.append(fast)

        best_argmax = np.argmax(sma_test_return)
        print(f'Best Slow Window: {slow_[best_argmax]}, Best Fast Window: {fast_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {sma_train_return[best_argmax]}, Best Testset Return: {sma_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (sma_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (sma_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (sma_train_return[best_argmax]) - 1 )/ len(train) * 30}, Best Testset Wealth/Month: {(10 ** (sma_test_return[best_argmax]) -1) / len(test) * 30}')
   
    def macd(self, sym, window_slow=26, window_fast=12, test_size=0.8, plot=False):
        def buy_sell(signal):
            Buy = []
            Sell = []
            flag = -1

            for i in range(0, len(signal)):
                if signal['MACD'].iloc[i] > signal['Signal'].iloc[i]:
                    Sell.append(np.nan)
                    if flag != 1:
                        Buy.append(signal['Close'].iloc[i])
                        flag = 1
                    else:
                        Buy.append(np.nan)
                elif signal['MACD'].iloc[i] < signal['Signal'].iloc[i]:
                    Buy.append(np.nan)
                    if flag != 0:
                        Sell.append(signal['Close'].iloc[i])
                        flag = 0
                    else:
                        Sell.append(np.nan)
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            return (Buy, Sell)
        
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]

        train['SlowEMA'] = train['Close'].ewm(span=window_slow, adjust=False).mean()
        train['FastEMA'] = train['Close'].ewm(span=window_fast, adjust=False).mean()

        train['MACD'] = train['FastEMA'] - train['SlowEMA']
        train['Signal'] = train['MACD'].ewm(span=9, adjust=False).mean()

        train['Buy'] = buy_sell(train)[0]
        train['Sell'] = buy_sell(train)[1]
        
        test.loc[:,'SlowEMA'] = test.loc[:,'Close'].ewm(span=window_slow, adjust=False).mean()
        test.loc[:,'FastEMA'] = test.loc[:,'Close'].ewm(span=window_fast, adjust=False).mean()

        test['MACD'] = test['FastEMA'] - test['SlowEMA']
        test['Signal'] = test['MACD'].ewm(span=9, adjust=False).mean()

        test['Buy'] = buy_sell(test)[0]
        test['Sell'] = buy_sell(test)[1]
        
        is_invested = False
        train.loc[:,'IsInvested'] = np.where(train.loc[:,'Sell'] >= 0, False, True)
        train.loc[:,'AlgoLogReturn'] = train.loc[:,'IsInvested'] * train.loc[:,'log_ret']
        
        is_invested = False
        test.loc[:,'IsInvested'] = np.where(test.loc[:,'Sell'] >= 0, False, True)
        test.loc[:,'AlgoLogReturn'] = test.loc[:,'IsInvested'] * test.loc[:,'log_ret']
        
        self.algoreturn_train_MACD = train.loc[train.index[:-1],'AlgoLogReturn'].sum()
        self.algoreturn_test_MACD = test.loc[test.index[:-1],'AlgoLogReturn'].sum()
        
        if plot == True:
            fig = go.Figure(data=[
                go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'], 
                close=train['Close'],
                name='Candlestick')])

            # Create and style traces
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['SlowEMA'], 
                                     name = 'SlowEMA (train)',
                                     line=dict(color='red', width=1)))

            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['FastEMA'],
                                     name='FastEMA (train)',
                                     line=dict(
                                         color='blue', 
                                         width=1))) # dash options include 'dash', 'dot', and 'dashdot'


            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3))) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)))  
            
            # Edit the layout    
            fig.update_layout(
                xaxis_rangeslider_visible=False, 
                title='Trainset',
                xaxis_title='Date',
                yaxis_title='USD ($)')
            fig.show()
            
            fig2 = go.Figure(data=[
                go.Candlestick(x=test['Date'],
                open=test['Open'],
                high=test['High'],
                low=test['Low'], 
                close=test['Close'],
                name='Candlestick')])
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3))) 

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)))  
            
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['SlowEMA'], 
                                     name = 'SlowEMA (Test)',
                                     line=dict(
                                         color='red', 
                                         width=1)))

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['FastEMA'],
                                     name='FastEMA (Test)',
                                     line=dict(
                                         color='blue', 
                                         width=1))) # dash options include 'dash', 'dot', and 'dashdot'

            # Edit the layout    
            fig2.update_layout(
                xaxis_rangeslider_visible=False, 
                title='Testset',
                xaxis_title='Date',
                yaxis_title='USD ($)')
            fig2.show()

        return self.algoreturn_train_MACD, self.algoreturn_test_MACD
    
    def macd_gs(self, sym, search_para_slow=(26, 51), search_para_fast=(1, 26)):
        macd_train_return = []
        macd_test_return = []
        slow_ = []
        fast_ = []
        for slow in range(search_para_slow[0], search_para_slow[1]):
            for fast in range(search_para_fast[0], search_para_fast[1]):
                macd_train_return.append(fin.macd(sym, window_slow=slow, window_fast=fast, test_size=0.8)[0])
                macd_test_return.append(fin.macd(sym, window_slow=slow, window_fast=fast, test_size=0.8)[1])
                slow_.append(slow)
                fast_.append(fast)

        best_argmax = np.argmax(macd_test_return)
        print(f'Best Slow Window: {slow_[best_argmax]}, Best Fast Window: {fast_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {macd_train_return[best_argmax]}, Best Testset Return: {macd_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (macd_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (macd_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (macd_train_return[best_argmax]) - 1 )/ len(train) * 30}, Best Testset Wealth/Month: {(10 ** (macd_test_return[best_argmax]) - 1) / len(test) * 30}') 
    
    def arima(self, sym):
        # Price Simulation
        arima_Ntest = 30
        arima_train = self.stocks[self.stocks['sym'] == sym]['Adj Close'].iloc[:-arima_Ntest]
        arima_test = self.stocks[self.stocks['sym'] == sym]['Adj Close'].iloc[-arima_Ntest:]
        arima = pm.auto_arima(
            arima_train,
            error_action='ignore',
            trace=True,
            suppress_warnings=True, 
            maxiter=10,
            seasonal=False)

        arima_params = arima.get_params()
        arima_d = arima_params['order'][1]

        arima_train_pred = arima.predict_in_sample(start=arima_d, end=-1)
        arima_test_pred, confint = arima.predict(n_periods=arima_Ntest, return_conf_int=True)
        
        x = self.stocks[self.stocks['sym'] == sym][-arima_Ntest:]['Date']
        x_rev = x[::-1]

        y1_upper = confint[:,1]
        y1_lower = confint[:,0]
        y1_lower = y1_lower[::-1]
        
        fig = go.Figure(data=[go.Candlestick(x=self.stocks[self.stocks['sym'] == sym]['Date'],
                open=self.stocks[self.stocks['sym'] == sym]['Open'],
                high=self.stocks[self.stocks['sym'] == sym]['High'],
                low=self.stocks[self.stocks['sym'] == sym]['Low'],
                close=self.stocks[self.stocks['sym'] == sym]['Close'])])
        
        fig.add_trace(go.Scatter(
                name='Forecast',
                x=self.stocks[self.stocks['sym'] == sym][-arima_Ntest:]['Date'],
                y=arima_test_pred,
                mode='lines'))
        
        fig.add_trace(go.Scatter(
                name='Upper Bound',
                x=x,
                y=y1_upper,
                mode='lines',
                line=dict(width=1),
                fill='tonexty'))
        
        fig.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_rev,
                y=y1_lower,
                mode='lines',
                line=dict(width=1),
                fill='tonexty'))

        fig.update_layout(yaxis_title='USD ($)')
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()
  
    def HMM(self, sym):
        returns = self.returns.loc[:, sym]
        returns.dropna(inplace=True)
        model = hmm.GaussianHMM(n_components=2, covariance_type='diag')
        X = returns.to_numpy().reshape(-1, 1)
        model.fit(X) # Viterbi Algo is used to find the max proba, mean and variance
        Z = model.predict(X)
        
        # we want to draw different segments in different colors according to state
        fig, ax = plt.subplots(figsize=(10, 5))

        # first create arrays with nan
        Z2 = pd.DataFrame(Z, index=returns.index, columns=['state'])

        self.stocks[self.stocks['sym'] == sym].loc[1:, 'state'] = Z2.replace({0:'low volatility', 1:'high volatility'})

        returns0 = np.empty(len(Z))
        returns1 = np.empty(len(Z))
        returns0[:] = np.nan
        returns1[:] = np.nan

        # fill in the values only if the state is the one correspoding to the array
        returns0[Z == 0] = returns[Z == 0]
        returns1[Z == 1] = returns[Z == 1]
        
        plt.plot(returns.index, returns0, label='Low Volatility')
        plt.plot(returns.index, returns1, label='High Volatility')
        plt.legend()
        
    def bollinger(self, sym, window=20, m=2, plot=False):
        period = window
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]

        train.loc[:, 'Typical Price'] = (train['High'] + train['Low'] + train['Close'])/3

        train.loc[:, 'Bollinger_SMA'] = train.loc[:, 'Typical Price'].rolling(window=window).mean()
        train.loc[:, 'Bollinger_STD'] = train.loc[:, 'Typical Price'].rolling(window=window).std()

        train.loc[:, 'Bollinger_Upper'] = train['Bollinger_SMA'] + train['Bollinger_STD'] * m
        train.loc[:, 'Bollinger_Lower'] = train['Bollinger_SMA'] - train['Bollinger_STD'] * m

        train['Bollinger_Sell'] = np.where(train['Close'] >= train['Bollinger_Upper'], 1, np.nan) * train['Close']
        train['Bollinger_Buy'] = np.where(train['Close'] <= train['Bollinger_Lower'], 1, np.nan) * train['Close']

        train.loc[:,'IsInvested_bollinger'] = np.where(train.loc[:, 'Bollinger_Sell'] >= 0, False, True)

        train.loc[:,'log_ret_bollinger'] = train.loc[:,'IsInvested_bollinger'] * train.loc[:,'log_ret']
        

        test.loc[:, 'Typical Price'] = (test['High'] + test['Low'] + test['Close'])/3

        test.loc[:, 'Bollinger_SMA'] = test.loc[:, 'Typical Price'].rolling(window=window).mean()
        test.loc[:, 'Bollinger_STD'] = test.loc[:, 'Typical Price'].rolling(window=window).std()

        test.loc[:, 'Bollinger_Upper'] = test.loc[:, 'Bollinger_SMA'] + test.loc[:, 'Bollinger_STD'] * m
        test.loc[:, 'Bollinger_Lower'] = test.loc[:, 'Bollinger_SMA'] - test.loc[:, 'Bollinger_STD'] * m

        test['Bollinger_Sell'] = np.where(test['Close'] >= test['Bollinger_Upper'], 1, np.nan) * test['Close']
        test['Bollinger_Buy'] = np.where(test['Close'] <= test['Bollinger_Lower'], 1, np.nan) * test['Close']

        test.loc[:,'IsInvested_bollinger'] = np.where(test.loc[:, 'Bollinger_Sell'] >= 0, False, True)

        test.loc[:,'log_ret_bollinger'] = test.loc[:,'IsInvested_bollinger'] * test.loc[:,'log_ret']


        self.algoreturn_train_bollinger = train.loc[:,'log_ret_bollinger'].sum()
        self.algoreturn_test_bollinger = test.loc[:,'log_ret_bollinger'].sum()
        
        if plot == True:
            x = train['Date']
            x_rev = x[::-1]

            y_upper = train['Bollinger_Upper']
            y_lower = train['Bollinger_Lower']
            
            fig = go.Figure(data=[
                go.Candlestick(x=x,
                open=train['Open'],
                high=train['High'],
                low=train['Low'], 
                close=train['Close'],
                name='Candlestick'),
            
                go.Scatter(
                    name='Bollinger Upper',
                    x=train['Date'],
                    y=y_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=True
                ),
                go.Scatter(
                    name='Bollinger Upper',
                    x=train['Date'],
                    y=y_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ),
                go.Scatter(x=train['Date'],
                                     y=train['Bollinger_Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                     color='#62a0cb',
                                     width=3)
                ), 
                go.Scatter(x=train['Date'],
                                     y=train['Bollinger_Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                     color='#cb62a0',
                                     width=3)
                )
                
            ])

            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.update_layout(title='Trainset')
            fig.show()
            
            x2 = test['Date']
            x2_rev = x[::-1]

            y2_upper = test['Bollinger_Upper']
            y2_lower = test['Bollinger_Lower']
            
            fig2 = go.Figure(data=[
                go.Candlestick(x=x2,
                open=test['Open'],
                high=test['High'],
                low=test['Low'], 
                close=test['Close'],
                name='Candlestick'),
            
                go.Scatter(
                    name='Bollinger Upper',
                    x=test['Date'],
                    y=y2_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=True
                ),
                go.Scatter(
                    name='Bollinger Upper',
                    x=test['Date'],
                    y=y2_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ),
                go.Scatter(x=test['Date'],
                                     y=test['Bollinger_Buy'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                     color='#62a0cb',
                                     width=3)
                ), 
                go.Scatter(x=test['Date'],
                                     y=test['Bollinger_Sell'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                     color='#cb62a0',
                                     width=3)
                )
                
            ])

            fig2.update_layout(yaxis_title='USD ($)')
            fig2.update_layout(xaxis_rangeslider_visible=False)
            fig2.update_layout(title='testset')
            fig2.show()

        return self.algoreturn_train_bollinger, self.algoreturn_test_bollinger
    
    def bollinger_gs(self, sym, search_para_window=(7, 51), search_para_m=3):
        bollinger_train_return = []
        bollinger_test_return = []
        m_ = []
        window_ = []
        for m in range(1, search_para_m):
            for window in range(search_para_window[0], search_para_window[1]):
                bollinger_train_return.append(fin.Bollinger(sym, window=window, m=m)[0])
                bollinger_test_return.append(fin.Bollinger(sym, window=window, m=m)[1])
                m_.append(m)
                window_.append(window)

        best_argmax = np.argmax(bollinger_test_return)
        print(f'Best Window: {window_[best_argmax]}, Best m: {m_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {bollinger_train_return[best_argmax]}, Best Testset Return: {bollinger_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (bollinger_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (bollinger_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (bollinger_train_return[best_argmax]) - 1 )/ len(train) * 30}, Best Testset Wealth/Month: {(10 ** (bollinger_test_return[best_argmax]) - 1) / len(test) * 30}') 
    
    def rsi(self, sym, window=14, threshold_upper=70, threshold_lower=30, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.returns[sym].iloc[:-Ntest]
        test = self.returns[sym].iloc[-Ntest:]
        
        avg_gain_train = pd.DataFrame(np.where(train > 0, train, 0), columns=['avg_gain_train']).rolling(14).mean()
        avg_loss_train = pd.DataFrame(np.where(train < 0, abs(train), 0), columns=['avg_loss_train']).rolling(14).mean()
        RSI_train = 100 - (100/(1 + avg_gain_train['avg_gain_train']/avg_loss_train['avg_loss_train']))
        
        avg_gain_test = pd.DataFrame(np.where(test > 0, test, 0), columns=['avg_gain_test']).rolling(14).mean()
        avg_loss_test = pd.DataFrame(np.where(test < 0, abs(test), 0), columns=['avg_loss_test']).rolling(14).mean()
        RSI_test = 100 - (100/(1 + avg_gain_test['avg_gain_test']/avg_loss_test['avg_loss_test']))

        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]
        train.loc[:, 'RSI'] = RSI_train
        test.loc[:, 'RSI'] = RSI_test
        
        train.loc[:, 'Sell_RSI'] = np.nan
        train.loc[:, 'Buy_RSI'] = np.nan
        test.loc[:, 'Sell_RSI'] = np.nan
        test.loc[:, 'Buy_RSI'] = np.nan
        
        train['Sell_RSI'][1:] = np.where(RSI_train > threshold_upper, train['Close'][1:], np.nan)
        train['Buy_RSI'][1:] = np.where(RSI_train < threshold_lower, train['Close'][1:], np.nan)
        
        test['Sell_RSI'] = np.where(RSI_test > threshold_upper, test['Close'], np.nan)
        test['Buy_RSI'] = np.where(RSI_test < threshold_lower, test['Close'], np.nan)
        
        train['Algo_return'] = np.where(train['Sell_RSI'][1:] > 0, False, True) * train['log_ret'][1:]
        test['Algo_return'] = np.where(test['Sell_RSI'] > 0, False, True) * test['log_ret']
        
        self.Algo_return_train_rsi = train['Algo_return'].sum()
        self.Algo_return_test_rsi = test['Algo_return'].sum()
        
        if plot == True:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Trainset Close Price", "Trainset Relative Strength Index"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig.add_trace(go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'],
                close=train['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=RSI_train,
                                     mode='lines',
                                     name='RSI', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig.add_shape(type="line",
                            x0=train['Date'].min(), 
                            y0=threshold_upper, 
                            x1=train['Date'].max(),
                            y1=threshold_upper,
                            line=dict(
                                color="LightSeaGreen",
                                width=2,
                                dash="dashdot"),
                            row=2, 
                            col=1)
            
            fig.add_shape(type="line",
                            x0=train['Date'].min(), 
                            y0=threshold_lower, 
                            x1=train['Date'].max(),
                            y1=threshold_lower,
                            line=dict(
                                color="red",
                                width=2,
                                dash="dashdot"),
                            row=2, 
                            col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy_RSI'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell_RSI'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            
            fig.update_layout(title='Trainset')
            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()
            
            fig2 = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Testset Close Price", "Testset Relative Strength Index"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig2.add_trace(go.Candlestick(x=test['Date'],
                open=test['Open'],
                high=test['High'],
                low=test['Low'],
                close=test['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'], 
                                     y=RSI_test,
                                     mode='lines',
                                     name='RSI', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig2.add_shape(type="line",
                            x0=test['Date'].min(), 
                            y0=threshold_upper, 
                            x1=test['Date'].max(),
                            y1=threshold_upper,
                            line=dict(
                                color="LightSeaGreen",
                                width=2,
                                dash="dashdot"),
                            row=2, 
                            col=1)
            
            fig2.add_shape(type="line",
                            x0=test['Date'].min(), 
                            y0=threshold_lower, 
                            x1=test['Date'].max(),
                            y1=threshold_lower,
                            line=dict(
                                color="red",
                                width=2,
                                dash="dashdot"),
                            row=2, 
                            col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Buy_RSI'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Sell_RSI'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            
            fig2.update_layout(title='testset')
            fig2.update_layout(yaxis_title='USD ($)')
            fig2.update_layout(xaxis_rangeslider_visible=False)
            fig2.show()

        return self.Algo_return_train_rsi, self.Algo_return_test_rsi
    
    def rsi_gs(self, sym, search_para_upper=(70, 100), search_para_window=(14, 54)):
        rsi_train_return = []
        rsi_test_return = []
        window_ = []
        upper_ = []
        for w in range(search_para_window[0], search_para_window[1]):
            for u in range(search_para_upper[0], search_para_upper[1]):
                rsi_train_return.append(fin.rsi(sym, threshold_upper=u, threshold_lower=100-u, window=w)[0])
                rsi_test_return.append(fin.rsi(sym, threshold_upper=u, threshold_lower=100-u, window=w)[1])
                window_.append(w)
                upper_.append(u)
                
        best_argmax = np.argmax(rsi_test_return)
        print(f'Best Window: {window_[best_argmax]}, Best Boundary: {upper_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {rsi_train_return[best_argmax]}, Best Testset Return: {rsi_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (rsi_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (rsi_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (rsi_train_return[best_argmax]) - 1 )/ len(train) * 30}, Best Testset Wealth/Month: {(10 ** (rsi_test_return[best_argmax]) - 1) / len(test) * 30}')
    
    def obv(self, sym, window=20, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]
        
        OBV = []
        OBV.append(0)

        for i in range(1, len(train['Close'])):
            if train.Close.iloc[i] > train.Close.iloc[i-1]:
                OBV.append(OBV[-1] + train.Volume.iloc[i])
            elif train.Close.iloc[i] < train.Close.iloc[i-1]:
                OBV.append(OBV[-1] - train.Volume.iloc[i])
            else:
                OBV.append(OBV[-1])

        train['OBV'] = OBV
        train['OBV_EMA'] = train['OBV'].ewm(span=window).mean()
        train['Buy_OBV'] = np.where(train['OBV'] > train['OBV_EMA'], train['Close'], np.nan)
        train['Sell_OBV'] = np.where(train['OBV'] < train['OBV_EMA'], train['Close'], np.nan)

        train['Is_invested'] = np.where(train['Sell_OBV'] > 0, False, True)
        train['Alog_return'] = train['Is_invested'] * train['log_ret']
        self.Algo_return_train_obv = train['Alog_return'].sum()
        
        if plot == True:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Trainset Close Price", "Trainset On-Balance Volume"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig.add_trace(go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'],
                close=train['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['OBV'],
                                     mode='lines',
                                     name='OBV', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy_OBV'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell_OBV'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            
            fig.update_layout(title='Trainset')
            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()
    
        OBV = []
        OBV.append(0)

        for i in range(1, len(test['Close'])):
            if test.Close.iloc[i] > test.Close.iloc[i-1]:
                OBV.append(OBV[-1] + test.Volume.iloc[i])
            elif test.Close.iloc[i] < test.Close.iloc[i-1]:
                OBV.append(OBV[-1] - test.Volume.iloc[i])
            else:
                OBV.append(OBV[-1])

        test['OBV'] = OBV
        test['OBV_EMA'] = test['OBV'].ewm(span=window).mean()
        test['Buy_OBV'] = np.where(test['OBV'] > test['OBV_EMA'], test['Close'], np.nan)
        test['Sell_OBV'] = np.where(test['OBV'] < test['OBV_EMA'], test['Close'], np.nan)

        test['Is_invested'] = np.where(test['Sell_OBV'] > 0, False, True)
        test['Alog_return'] = test['Is_invested'] * test['log_ret']
        self.Algo_return_test_obv = test['Alog_return'].sum()
        
        if plot == True:
            fig2 = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Testset Close Price", "Testset On-Balance Volume"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig2.add_trace(go.Candlestick(x=test['Date'],
                open=test['Open'],
                high=test['High'],
                low=test['Low'],
                close=test['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'], 
                                     y=test['OBV'],
                                     mode='lines',
                                     name='OBV', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Buy_OBV'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Sell_OBV'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            
            fig2.update_layout(title='testset')
            fig2.update_layout(yaxis_title='USD ($)')
            fig2.update_layout(xaxis_rangeslider_visible=False)
            fig2.show()
        
        return self.Algo_return_train_obv, self.Algo_return_test_obv
    
    def obv_gs(self, sym, search_param=60):
        obv_train_return = []
        obv_test_return = []
        window_ = []
        for w in range(1, search_param+1):
            obv_res = self.obv(sym, window=w, plot=False)
            obv_train_return.append(obv_res[0])
            obv_test_return.append(obv_res[1])    
            window_.append(w)
            
        best_argmax = np.argmax(obv_test_return)
        print(f'Best Window: {window_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {obv_train_return[best_argmax]}, Best Testset Return: {obv_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (obv_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (obv_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (obv_train_return[best_argmax]) - 1 )/ 2216 * 30}, Best Testset Wealth/Month: {(10 ** (obv_test_return[best_argmax]) - 1) / 554 * 30}')
    
    def ikh(self, sym, Tenkan_window=9, Kijun_window=26, SenkouB_window=52, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]
        
        # Tenkan-Sen
        train['Tenkan-Sen'] = (train['High'].rolling(Tenkan_window).max() + train['Low'].rolling(Tenkan_window).min())/2
        
        # Kijun-Sen
        train['Kijun-Sen'] = (train['High'].rolling(Kijun_window).max() + train['Low'].rolling(Kijun_window).min())/2
        
        # Chikou Span
        train['Chikou Span'] = train['Close'].shift(-26)
        
        # Senkou A
        train['Senkou A'] = ((train['Tenkan-Sen'] + train['Kijun-Sen'])/2).shift(26)
        
        # Senkou B
        train['Senkou B'] = ((train['High'].rolling(SenkouB_window).max() + train['Low'].rolling(SenkouB_window).min())/2).shift(52)
        
        train['shade'] = np.where(train['Senkou A'] >= train['Senkou B'], 1, 0)
        
        train['Buy_ikh'] = np.where(train['Senkou A'] >= train['Senkou B'], train['Close'], 0)
        train['Sell_ikh'] = np.where(train['Senkou A'] <= train['Senkou B'], train['Close'], 0)
        train['Is_invested'] = np.where(train['Sell_ikh'] > 0, False, True)
        train['Algo_return'] = train['Is_invested'] * train['log_ret']
        self.Algo_return_train_ikh = train['Algo_return'].sum()
        
        if plot == True:
            fig = go.Figure(data=[go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'],
                close=train['Close'],
                name='Candlestick')])
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['Tenkan-Sen'],
                                     mode='lines',
                                     name='Tenkan-Sen', 
                                     line=dict(color='#00B5F7')))
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['Kijun-Sen'],
                                     mode='lines',
                                     name='Kijun-Sen', 
                                     line=dict(color='#D62728')))
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['Chikou Span'],
                                     mode='lines',
                                     name='Chikou Span', 
                                     line=dict(color='black')))
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=np.where(train['Senkou A'] > train['Senkou B'], train['Senkou A'], np.nan),
                                     mode='lines',
                                     name='Senkou A', 
                                     line=dict(color='#16FF32'))) 
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=np.where(train['Senkou A'] < train['Senkou B'], train['Senkou B'], np.nan),
                                     mode='lines',
                                     name='Senkou B', 
                                     line=dict(color='#F6222E'), 
                                     fill='tonexty', 
                                     fillcolor = 'rgba(250,0,0,0.4)'))
            
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy_ikh'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3))) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell_ikh'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)))
            
            
            fig.update_layout(title='Trainset')
            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()
            
        # Tenkan-Sen
        test['Tenkan-Sen'] = (test['High'].rolling(9).max() + test['Low'].rolling(9).min())/2
        
        # Kijun-Sen
        test['Kijun-Sen'] = (test['High'].rolling(26).max() + test['Low'].rolling(26).min())/2
        
        # Chikou Span
        test['Chikou Span'] = test['Close'].shift(-26)
        
        # Senkou A
        test['Senkou A'] = (test['Tenkan-Sen'] + test['Kijun-Sen'])/2
        
        # Senkou B
        test['Senkou B'] = (test['High'].rolling(52).max() + test['Low'].rolling(52).min())/2
        
        test['shade'] = np.where(test['Senkou A'] > test['Senkou B'], 1, 0)
        
        test['Buy_ikh'] = np.where(test['Senkou A'] >= test['Senkou B'], test['Close'], 0)
        test['Sell_ikh'] = np.where(test['Senkou A'] <= test['Senkou B'], test['Close'], 0)
        test['Is_invested'] = np.where(test['Sell_ikh'] > 0, False, True)
        test['Algo_return'] = test['Is_invested'] * test['log_ret']
        self.Algo_return_test_ikh = test['Algo_return'].sum()
        
        return self.Algo_return_train_ikh, self.Algo_return_test_ikh
        
    def so(self, sym, window_low=14, window_high=14, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]
        
        C = train['Close'].shift(1)
        L14 = train['Low'].rolling(window_low).min()
        H14 = train['High'].rolling(window_high).max()
        K = ((C - L14)/(H14 - L14))*100
        
        train['Stochastic Oscillator'] = K
        train['Buy_SO'] = np.where(train['Stochastic Oscillator'] < 20, train['Close'], np.nan)
        train['Sell_SO'] = np.where(train['Stochastic Oscillator'] > 80, train['Close'], np.nan)
        
        train['Is_invested'] = np.where(train['Sell_SO'] > 0, False, True)
        train['Algo_return_SO'] = train['Is_invested'] * train['log_ret']
        
        self.Algo_return_so_train = train['Algo_return_SO'].sum()
        
        if plot == True:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Trainset Close Price", "Trainset Stochastic Oscillator"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig.add_trace(go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'],
                close=train['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['Stochastic Oscillator'],
                                     mode='lines',
                                     name='OBV', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy_SO'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell_SO'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            fig.add_shape(type="rect",
                x0=train['Date'].min(), y0=20, x1=train['Date'].max(), y1=80,
                line=dict(color="LightSkyBlue", width=1,),
                fillcolor="LightSalmon",
                row=2, 
                col=1, 
                opacity=0.5, 
                line_width=0)
            
            fig.update_layout(title='Trainset')
            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()
            
        C = test['Close'].shift(1)
        L14 = test['Low'].rolling(window_low).min()
        H14 = test['High'].rolling(window_high).max()
        K = ((C - L14)/(H14 - L14))*100
        
        test['Stochastic Oscillator'] = K
        test['Buy_SO'] = np.where(test['Stochastic Oscillator'] < 20, test['Close'], np.nan)
        test['Sell_SO'] = np.where(test['Stochastic Oscillator'] > 80, test['Close'], np.nan)
        
        test['Is_invested'] = np.where(test['Sell_SO'] > 0, False, True)
        test['Algo_return_SO'] = test['Is_invested'] * test['log_ret']
        
        self.Algo_return_so_test = test['Algo_return_SO'].sum()
        
        if plot == True:
            fig2 = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Testset Close Price", "Testset Stochastic Oscillator"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig2.add_trace(go.Candlestick(x=test['Date'],
                open=test['Open'],
                high=test['High'],
                low=test['Low'],
                close=test['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'], 
                                     y=test['Stochastic Oscillator'],
                                     mode='lines',
                                     name='OBV', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Buy_SO'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (Test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Sell_SO'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (Test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            fig2.add_shape(type="rect",
                x0=test['Date'].min(), y0=20, x1=test['Date'].max(), y1=80,
                line=dict(color="LightSkyBlue", width=1,),
                fillcolor="LightSalmon",
                row=2, 
                col=1, 
                opacity=0.5,
                line_width=0)
            
            fig2.update_layout(title='Testset')
            fig2.update_layout(yaxis_title='USD ($)')
            fig2.update_layout(xaxis_rangeslider_visible=False)
            fig2.show()
            
        return self.Algo_return_so_train, self.Algo_return_so_test
    
    def so_gs(self, sym, search_param_low=(1, 42), search_param_high=(1, 42)):
        so_train_return = []
        so_test_return = []
        window_low_ = []
        window_high_ = []
        for l in range(search_param_high[0], search_param_high[1]+1):
            for h in range(search_param_low[0], search_param_low[1]+1):
                so_res = self.so(sym, window_low=l, window_high=h, plot=False)
                so_train_return.append(so_res[0])
                so_test_return.append(so_res[1])    
                window_low_.append(l)
                window_high_.append(h)
            
        best_argmax = np.argmax(so_test_return)
        print(f'Best Window for High Price: {window_high_[best_argmax]}, Best Window for High Price: {window_low_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {so_train_return[best_argmax]}, Best Testset Return: {so_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (so_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (so_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (so_train_return[best_argmax]) - 1 )/ 2216 * 30}, Best Testset Wealth/Month: {(10 ** (so_test_return[best_argmax]) - 1) / 554 * 30}')
        
    def mfi(self, sym, threshold_upper=80, threshold_lower=20, window=14, plot=False):
        Ntest = int(round(len(self.stocks[self.stocks['sym'] == sym]) * (1 - 0.8), 1))
        train = self.stocks[self.stocks['sym'] == sym].iloc[:-Ntest]
        test = self.stocks[self.stocks['sym'] == sym].iloc[-Ntest:]
        
        TypicalPrice = (train['High'] + train['Low'] + train['Close'])/3
        RawMoneyFlow = TypicalPrice * train['Volume']
        MoneyFlowRatio = pd.Series(np.where(RawMoneyFlow.diff(1) > 0, RawMoneyFlow, 0)).rolling(14).sum()/pd.Series(np.where(RawMoneyFlow.diff(1) < 0, RawMoneyFlow, 0)).rolling(window).sum()
        MFI = 100 - 100/(1 + MoneyFlowRatio)
        
        train['MFI'] = MFI.to_list()
        train['Buy_MFI'] = np.where(train['MFI'] > threshold_upper, train['Close'], np.nan)
        train['Sell_MFI'] = np.where(train['MFI'] < threshold_lower, train['Close'], np.nan)
        train['Is_invested'] = np.where(train['Sell_MFI'] > 0, False, True)
        train['Algo_return_MFI'] = train['Is_invested'] * train['log_ret']
        self.Algo_return_MFI_train = train['Algo_return_MFI'].sum()
        
        if plot == True:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Trainset Close Price", "Trainset Money Flow Index"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig.add_trace(go.Candlestick(x=train['Date'],
                open=train['Open'],
                high=train['High'],
                low=train['Low'],
                close=train['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'], 
                                     y=train['MFI'],
                                     mode='lines',
                                     name='MFI', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Buy_MFI'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (train)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig.add_trace(go.Scatter(x=train['Date'],
                                     y=train['Sell_MFI'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (train)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            fig.add_shape(type="rect",
                x0=train['Date'].min(), y0=threshold_lower, x1=train['Date'].max(), y1=threshold_upper,
                line=dict(color="LightSkyBlue", width=1,),
                fillcolor="LightSalmon",
                row=2, 
                col=1, 
                opacity=0.5, 
                line_width=0)
            
            fig.update_layout(title='Trainset')
            fig.update_layout(yaxis_title='USD ($)')
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.show()
            
        TypicalPrice = (test['High'] + test['Low'] + test['Close'])/3
        RawMoneyFlow = TypicalPrice * test['Volume']
        MoneyFlowRatio = pd.Series(np.where(RawMoneyFlow.diff(1) > 0, RawMoneyFlow, 0)).rolling(14).sum()/pd.Series(np.where(RawMoneyFlow.diff(1) < 0, RawMoneyFlow, 0)).rolling(window).sum()
        MFI = 100 - 100/(1 + MoneyFlowRatio)
        
        test['MFI'] = MFI.to_list()
        test['Buy_MFI'] = np.where(test['MFI'] > threshold_upper, test['Close'], np.nan)
        test['Sell_MFI'] = np.where(test['MFI'] < threshold_lower, test['Close'], np.nan)
        test['Is_invested'] = np.where(test['Sell_MFI'] > 0, False, True)
        test['Algo_return_MFI'] = test['Is_invested'] * test['log_ret']
        self.Algo_return_MFI_test = test['Algo_return_MFI'].sum()
        
        if plot == True:
            fig2 = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Testset Close Price", "Testset Money Flow Index"),
                shared_xaxes=True,
                vertical_spacing =0.3)# tune this value until the two charts don't overlap
            
            fig2.add_trace(go.Candlestick(x=test['Date'],
                open=test['Open'],
                high=test['High'],
                low=test['Low'],
                close=test['Close'],
                name='Candlestick'),
                row=1, 
                col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'], 
                                     y=test['MFI'],
                                     mode='lines',
                                     name='MFI', 
                                     line=dict(color='#00B5F7')),
                                     row=2, 
                                     col=1)
            
            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Buy_MFI'],
                                     mode='markers',
                                     marker_symbol='triangle-up',
                                     name = 'Buy (test)',
                                     line=dict(
                                         color='#62a0cb',
                                         width=3)),
                                         row=1, 
                                         col=1) 

            fig2.add_trace(go.Scatter(x=test['Date'],
                                     y=test['Sell_MFI'],
                                     mode='markers',
                                     marker_symbol='triangle-down',
                                     name = 'Sell (test)',
                                     line=dict(
                                         color='#cb62a0',
                                         width=3)),
                                      row=1, 
                                      col=1) 
            
            fig2.add_shape(type="rect",
                x0=test['Date'].min(), y0=threshold_lower, x1=test['Date'].max(), y1=threshold_upper,
                line=dict(color="LightSkyBlue", width=1,),
                fillcolor="LightSalmon",
                row=2, 
                col=1, 
                opacity=0.5, 
                line_width=0)
            
            fig2.update_layout(title='Testset')
            fig2.update_layout(yaxis_title='USD ($)')
            fig2.update_layout(xaxis_rangeslider_visible=False)
            fig2.show()
            
        return self.Algo_return_MFI_train, self.Algo_return_MFI_test
    
    def mfi_gs(self, sym, search_param=42):
        mfi_train_return = []
        mfi_test_return = []
        window_ = []
        for w in range(1, search_param+1):
            mfi_res = self.mfi(sym, window=w, plot=False)
            mfi_train_return.append(mfi_res[0])
            mfi_test_return.append(mfi_res[1])    
            window_.append(w)
            
        best_argmax = np.argmax(mfi_test_return)
        print(f'Best Window: {window_[best_argmax]}')
        print(' ')
        print(f'Best Trainset Return: {mfi_train_return[best_argmax]}, Best Testset Return: {mfi_test_return[best_argmax]}')
        print(' ')
        print(f'Best Trainset Wealth: {10 ** (mfi_train_return[best_argmax]) - 1}, Best Testset Wealth: {10 ** (mfi_test_return[best_argmax]) - 1}')
        print(' ')
        print(f'Best Trainset Wealth/Month: {(10 ** (mfi_train_return[best_argmax]) - 1 )/ 2216 * 30}, Best Testset Wealth/Month: {(10 ** (mfi_test_return[best_argmax]) - 1) / 554 * 30}')
            
    def func(self):
        print('''
        1) stats(window=None)
        
        2) port_opt()
        
        3) plot_candles(sym)
        
        4) sma(sym, window_slow=200, window_fast=50, test_size=0.8, plot=False)
         a) sma_gs(sym, search_para_slow=(51, 200), search_para_fast=(1, 50))
        
        5) macd(sym, window_slow=26, window_fast=12, test_size=0.8, plot=False)
         a) macd_gs(sym, search_para_slow=(26, 51), search_para_fast=(1, 26))
        
        8) arima(sym)
        
        9) HMM(sym)
        
        10) bollinger(sym, window=20, m=2, plot=True)
         a) bollinger_gs(sym, search_para_window=(7, 51), search_para_m=3)       
        
        11) rsi(sym, window=14, threshold_upper=70, threshold_lower=30)
         a) rsi_gs(sym, search_para_upper=(70, 100), search_para_window=(14, 54))
        
        12) obv(sym, plot=False)
         a)  obv_gs(sym, search_param=60)
        
        13) ikh(sym, Tenkan_window=9, Kijun_window=26, SenkouB_window=52, plot=False)
        
        14) so(sym, plot=False)
         a) so_gs(sym, search_param_low=(1, 42), search_param_high=(1, 42))
        
        15) mfi(sym, threshold_upper=80, threshold_lower=20, window=14, plot=False)
         a) mfi_gs(sym, search_param=42)
        ''')

