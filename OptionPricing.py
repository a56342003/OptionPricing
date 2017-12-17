import numpy as np
import operator as op
import functools
from graphviz import Digraph
from scipy.stats import norm
import pandas as pd
#from sklearn.linear_model import LinearRegression

# 定義Combination函數
def ncr(n, r):
    r = min(r, n-r)                                     # 因nCr = nC(n-r) 因此使用較小的效率較好
    if r == 0: return 1                                 # 若r為0 則回傳1
    numer = functools.reduce(op.mul, range(n, n-r, -1)) # 由n連乘到n-r+1 即 n!/(n-r)!
    denom = functools.reduce(op.mul, range(1, r+1))     # 由1連乘到r     即 r!
    return numer//denom                                 # 回傳答案       即 n!/(n-r)!/r!


# 定義Option類別
class Option:

    # 初始化Option物件
    def __init__(self, S = 70, K = 130, r = 0.05, T = 1, sigma = 0.5, OptionType = 'Put', OptionStyle = 'American'):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.OptionType = OptionType
        self.OptionStyle = OptionStyle
        
    def BSModel(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T))
        d2 = d1-self.sigma * np.sqrt(self.T)
        if self.OptionType == 'Call':
            C = self.S * norm.cdf(d1) - np.exp(-self.r * self.T) * self.K * norm.cdf(d2)
            return C
        elif self.OptionType == 'Put':
            P = np.exp(-self.r * self.T) * self.K * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            return P
        else:
            print('請輸入正確的OptionType')
            return

    #二項式樹狀評價模型
    def binomialOptionPricing(self, N=10):

        # 先計算股價路徑所需的基本資訊
        deltat = self.T/N
        u = np.exp(self.sigma * np.sqrt(deltat))
        d = 1 / u
        a = np.exp(self.r * deltat)
        p = (a - d) / (u - d)
        
        # 歐式選擇權不能中途履約，因此直接使用期末股價折到期初即可
        if self.OptionStyle == 'European':

            # 計算到期股價            
            FinalStockPrice = np.ones(N + 1)
            for i in range(N+1):
                FinalStockPrice[i] = self.S * u**i * d**(N-i)

            # 計算買賣權到期價格
            if self.OptionType == 'Call':
                FinalStockPrice = FinalStockPrice - self.K
            elif self.OptionType == 'Put':
                FinalStockPrice = self.K - FinalStockPrice
            else:
                print('請輸入正確的OptionType')
                return 

            # 將價內的選擇權折現至期初，即可得到選擇權價格
            Price = 0            
            for i in range(N+1):
                if FinalStockPrice[i] > 0:
                    Price += ncr(N,i)* p**i * (1 - p)**(N-i) * FinalStockPrice[i]
                    
            return (Price / a**N)

        # 美式選擇權可中途履約，應檢視路徑中是否有提前履約的可能性
        elif self.OptionStyle == 'American':

            # 計算到期股價
            FinalStockPrice = np.ones(N + 1)
            for i in range(N+1):
                FinalStockPrice[i] = self.S * u**i * d**(N-i)

            # 計算買權到期價格
            if self.OptionType == 'Call':
                FinalStockPrice = FinalStockPrice - self.K
                FinalOptionPrice = np.amax([FinalStockPrice,np.zeros(len(FinalStockPrice))],axis=0)
                # 模擬買權路徑，並比較是否低於內涵價值
                for i in range(N - 1, -1, -1):
                    for j in range(i + 1):
                        FinalOptionPrice[j]=np.maximum(((1 - p) * FinalOptionPrice[j]+ p * FinalOptionPrice[j + 1])/a , self.S * u**j * d**(i - j) - self.K)
                # 折到期出即可得到結果
                return FinalOptionPrice[0]

            # 計算賣權到期價格
            elif self.OptionType == 'Put':
                FinalStockPrice = self.K - FinalStockPrice
                FinalOptionPrice = np.amax([FinalStockPrice,np.zeros(len(FinalStockPrice))],axis=0)
                # 模擬賣權路徑，並比較是否低於內涵價值
                for i in range(N - 1, -1, -1):
                    for j in range(i + 1):
                        FinalOptionPrice[j]=np.maximum(((1 - p) * FinalOptionPrice[j]+ p * FinalOptionPrice[j + 1])/a , self.K - self.S * u**j * d**(i - j))
               # 折到期出即可得到結果
                return FinalOptionPrice[0]

            else:
                print('請輸入正確的OptionType')
                return

        else:
            print('請輸入正確的OptionStyle')
            return

            
    # 畫出股價樹
    def drawStockTree(self, N=10):
        graph = Digraph(name = 'Stock Price Tree')
        deltat = self.T/N
        u = np.exp(self.sigma * np.sqrt(deltat))
        d = 1 / u
        a = np.exp(self.r * deltat)
        p = (a - d) / (u - d)
        
        # 計算出各節點之值並放入點中
        for i in range(N + 1):
            for j in range(i + 1):
                graph.node('S{},{}'.format(i,j),'{:0.2f}'.format(self.S * u**j * d**(i - j)))
                
        # 連接點跟點
        for i in range(N):
            for j in range(i + 1):
                graph.edge('S{},{}'.format(i,j),'S{},{}'.format(i + 1,j))
                graph.edge('S{},{}'.format(i,j),'S{},{}'.format(i + 1,j + 1))
        graph.format = 'png'
        graph.render('C:/Users/User/Desktop/PriceTree{}'.format(N))

    # 劃出選擇權樹
    def drawOptionTree(self, N=10):
        graph = Digraph(name = 'Call Price Tree')
        deltat = self.T/N
        u = np.exp(self.sigma * np.sqrt(deltat))
        d = 1 / u
        a = np.exp(self.r * deltat)
        p = (a - d) / (u - d)

        FinalStockPrice = np.ones(N + 1)
        for i in range(N+1):
            FinalStockPrice[i] = self.S * u**i * d**(N-i)
        if self.OptionType == 'Call':
            FinalStockPrice = FinalStockPrice - self.K
        elif self.OptionType == 'Put':
            FinalStockPrice = self.K - FinalStockPrice
        else:
            print('請輸入正確的OptionType')
            return
        
        FinalOptionPrice = np.amax([FinalStockPrice,np.zeros(len(FinalStockPrice))],axis=0)
        if self.OptionStyle == 'American':
            if self.OptionType == 'Call':
                for j in range(N + 1):
                    graph.node('S{},{}'.format(N,j),'{:0.2f}'.format(FinalOptionPrice[j]))
                for i in range(N - 1, -1 , -1):
                    for j in range(i + 1):
                        FinalOptionPrice[j]=np.maximum(((1 - p) * FinalOptionPrice[j]+ p * FinalOptionPrice[j + 1])/a ,self.S * u**j * d**(i - j) - self.K)
                        graph.node('S{},{}'.format(i,j),'{:0.2f}'.format(FinalOptionPrice[j]))
            elif self.OptionType == 'Put':
                for j in range(N + 1):
                    graph.node('S{},{}'.format(N,j),'{:0.2f}'.format(FinalOptionPrice[j]))
                for i in range(N - 1, -1 , -1):
                    for j in range(i + 1):
                        FinalOptionPrice[j]=np.maximum(((1 - p) * FinalOptionPrice[j]+ p * FinalOptionPrice[j + 1])/a , self.K - self.S * u**j * d**(i - j))
                        graph.node('S{},{}'.format(i,j),'{:0.2f}'.format(FinalOptionPrice[j]))
            else:
                print('請輸入正確的OptionType')
                return
            
        elif self.OptionStyle == 'European':

            for j in range(N + 1):
                    graph.node('S{},{}'.format(N,j),'{:0.2f}'.format(FinalOptionPrice[j]))
            for i in range(N - 1, -1 , -1):
                for j in range(i + 1):
                    FinalOptionPrice[j]=((1 - p) * FinalOptionPrice[j]+ p * FinalOptionPrice[j + 1])/a
                    graph.node('S{},{}'.format(i,j),'{:0.2f}'.format(FinalOptionPrice[j]))
        else:
            print('請輸入正確的OptionStyle')
            return
        
        # 連接點跟點
        for i in range(N):
            for j in range(i + 1):
                graph.edge('S{},{}'.format(i + 1,j),'S{},{}'.format(i,j))
                graph.edge('S{},{}'.format(i + 1,j + 1),'S{},{}'.format(i,j))
        graph.format = 'png'
        graph.render('C:/Users/User/Desktop/{}OptionTree{}'.format(self.OptionStyle,N))


    


            
    def priceSimulation(self, N = 5000,t=20):
        
        Price = self.S*np.ones(N)
        Price = np.c_[Price , Price * np.exp((self.r - 0.5 * self.sigma**2) * self.T / t + self.sigma * np.sqrt(self.T / t) * np.random.randn(N))]
                
#        for k in range(1,t):
#            Price = np.c_[Price , Price[:,k] * np.exp((self.r - 0.5 * self.sigma**2) * self.T / t + self.sigma * np.sqrt(self.T / t) * np.random.randn(N))]
        for k in range(0,t):
            Price = np.c_[Price , Price[:,k+1] * np.exp((self.r - 0.5 * self.sigma**2) * self.T / t + self.sigma * np.sqrt(self.T / t) * np.random.randn(N))]



        return Price


    def monteCarloSimulation(self, N = 5000,t=20):
        Prices = self.priceSimulation(N = N,t = t)

        if self.OptionStyle == 'European':
            if self.OptionType == 'Call':
                OptionT = np.maximum(np.zeros(N), Prices[:,t] - self.K)

            elif self.OptionType == 'Put':
                OptionT = np.maximum(np.zeros(N), self.K - Prices[:,t])

            else:
                print('OptionType should be Call or Put')
                return

            Option0 = OptionT*np.exp(-self.r*self.T)

            return np.mean(Option0)

        elif self.OptionStyle == 'American':
            if self.OptionType == 'Call':
                OptionT = np.maximum(np.zeros(N), Prices[:,t] - self.K)

            elif self.OptionType == 'Put':
                IntrinsicValue = np.maximum(np.zeros(Prices.shape), self.K - Prices)
                CashFlowTable = np.zeros(Prices.shape)
                CashFlowTable[:,t] = IntrinsicValue[:,t]

                #regr = LinearRegression()
                for i in range(t-1,0,-1):

                    Intrinsict = IntrinsicValue[:,i]
                    
                    CashFlowt = CashFlowTable[:,i+1]
                    Y = CashFlowt[Intrinsict>0] * np.exp(-self.r*self.T/t)
                    
                    St = Prices[:,i]
                    StintheMoney = St[Intrinsict>0]

                    #X = np.c_[np.ones(StintheMoney.shape), StintheMoney, StintheMoney**2]
                    X = np.c_[np.ones(StintheMoney.shape), np.exp(-StintheMoney/2),np.exp(-StintheMoney/2)*(1-StintheMoney),np.exp(-StintheMoney/2)*(1-2*StintheMoney+StintheMoney**2/2)]
                    Beta = np.linalg.lstsq(X,Y)[0]
                    ExpectedValueofContinuity = np.zeros(N)
                    ExpectedValueofContinuity[Intrinsict>0] = X.dot(Beta)
                    #X = np.c_[StintheMoney,StintheMoney**2]
                    #regr.fit(X,Y)
                    #ExpectedValueofContinuity = np.zeros(N)
                    #ExpectedValueofContinuity[Intrinsict>0] = regr.predict(X)

                    CashFlowTable[ExpectedValueofContinuity < Intrinsict] = 0
                    CashFlowTable[:,i][ExpectedValueofContinuity < Intrinsict] = Intrinsict[ExpectedValueofContinuity < Intrinsict]
                  #  print(CashFlowTable)
                  #  print('*'*100)


                for i in range(1,t+1):
                    CashFlowTable[:,i]=CashFlowTable[:,i]*np.exp(-self.r*self.T/t*i)
                    
                return np.sum(CashFlowTable)/N

            else:
                print('請輸入正確的OptionType')
                return

        else:
            print('請輸入正確的OptionStyle')
            return
                    
                    
                







