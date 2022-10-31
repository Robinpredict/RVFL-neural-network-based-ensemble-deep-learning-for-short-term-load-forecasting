
"""
This is a time series forecasting library. Its functionality includes the following three parts:
--------------------------------------------------Preprocessing------------------------

(a) difference:
input:  K -- order of differencing
output:  k_th order differenced series



(b) normalization or uniform
input: 0--normalization; 1--minmax
output: normalized(uniform) sequence



(c) denoising
input:
output:



(d) temporal-to-matrix (sliding window) returns
input: order k
output: (n-k)*k matrix , n-k vector


(e) wavelet information acquisition
input: wavelet name, decomposition level, "swt" or "dwt", padding mode ,window_length L

output: (n-L)* m    -- L is the window length, m is the number of wavelet information ,n is the length of sequence



--------------------------------------------------Temporal analysis of series ------------------------
(f) wavelet decomposition
input: wavelet name, decomposition level, "swt" or "dwt", padding mode ,window_length L
output:  every part of decomposition


(g) data-driven decomposition
input: choice of decomposition methods: CEEMD, EWT, VMD;  and the configuration of each method
output: every part of decomposition

(h) autocorr
input:
output:

(i) xcorr
input:
output:


--------------------------------------------------Benchmark univariate forecasting model ------------------------
(j) Naive
input:  whole time series, test_length
output: naive_hat


(k) SARIMA + AIC/BIC
input: whole time series, test_length
output: sarima_hat


(l) ES
input:
output:


(m)LR
input: whole time series, test_length, k
output: lr_hat


(o) SVM
input: whole time series, test_length, k
output: SVM_hat

(p) lightGBM
input: whole time series, test_length, k
output: lightGBM_hat

(q) MLP
input: whole time series, test_length, k
output: MLP_hat


(r) RF
input: whole time series, test_length, k
output: RF_hat


(s) CNN
input: whole time series, test_length, k
output: CNN_hat

(t) LSTM
input: whole time series, test_length, k
output: LSTM_hat


*latest new method
(u) prophet



(v) deepAR


(u) deepStateSpace


(w) TCN






--------------------------------------------------   Metric&Evaluation  ------------------------
(x) RMSE,MAE,MAPE


(y) MASE,


(Z)

"""

'''===========================================预处理====================================='''
import os
import sys
# curPath=os.path.abspath(os.path.dirname(__file__))
# rootPath=os.path.split(curPath)[0]
# sys.path.append(rootPath)
# par_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(par_dir)


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import numpy as np
import random

random.seed(53113)
np.random.seed(53113)
import torch
torch.manual_seed(53113)

class TsPrep(object):
    def __init__(self):
        pass

    def ts_diff(self,x):
        initial = x[0]
        diffed = np.diff(x)
        return initial, diffed

    def ts_normalize(self,x, way=1):
        # 1--normalization  0--minmax
        if way == 1:
            mean_x = np.mean(x)
            std_x = np.std(x)
            norm_x = (x - mean_x) / std_x
            return mean_x, std_x, norm_x

        if way == 0:
            min_x = np.min(x)
            max_x = np.max(x)
            scale_x = (x - min_x) / (max_x - min_x)
            return min_x, max_x, scale_x

    def ts_denoise(self,x):  # 暂时还不写，完成既定任务后再完善 TBC
        pass

    def ts_s2m(self,x, k):  # transform a time series(length n ) to a matrix (n-k,k) and a n-k long series
        ls = np.shape(x)[0]
        assert (ls > k) & (type(k) == int)
        # 按照(x_(t-k),x(t-(k-1)),...x_(t-1))的顺序组成
        m = np.empty([ls - k, k])

        for i in range(0, k):  # order = k, k-1,...,2,1
            col_i = x[i:i + ls - k]  # each column has fixed length(height) ls-k
            m[:, i] = col_i

        target = x[k:k + ls - k]  # order = 0

        return m, target

    def ts_walkforward(self,x,w,k,decompose,pad_l=0,pad_raw=False):
        """
        :param w:  window length
        :param k:  num of subseries
        :param decompose: vmd or dwt use db7 symmetrical
        :return: num_loc * k * w    num_loc = length(x) - w , last window isn't calculated
        #PAD RAW pad original data
        """
        TsAnalyzer_=TsAnalyzer()
        print('walk_forward')
        m_x,_ = self.ts_s2m(x,w)
        if pad_l == 0:
            temp1 = [m_x[i,:]  for i in range(len(m_x))]
        else:
            data=np.tile(m_x[:,-1],(pad_l,1)).T
            pad_x=np.concatenate((m_x,data),axis=1)
            temp1 = [pad_x[i,:]  for i in range(len(m_x))]
        print(decompose)
        if decompose == 'dwt':
            print('dwt')
            if pad_raw:
                print('ww')
                if pad_l:
                    pass
                else:
                    print('ww')
                    for i in temp1:
                        print(i.shape,TsAnalyzer_.ts_DWT(i,'db7','symmetric',k-1).shape)
#                   
                    return np.array([np.hstack((i.reshape(-1,1),TsAnalyzer_.ts_DWT(i,'db7','symmetric',k-1)))  for i in temp1])
            else:    
                return  np.array([TsAnalyzer_.ts_DWT(i,'db7','symmetric',k-1) for i in temp1])
        elif decompose == 'vmd':
            if pad_raw:
                if pad_l:
                    pass
                else:
#                    for i in temp1:
#                        print(i.shape,TsAnalyzer.ts_VMD(i,k)[0].shape)
#                   
                    return np.array([np.hstack((i.reshape(-1,1),TsAnalyzer_.ts_VMD(i,k=k)[0].T))  for i in temp1])
            else:
                if pad_l: 
                
                    return np.array([TsAnalyzer.ts_VMD(i,k=k)[0][:-pad_l,:]  for i in temp1])
                else:
#                    for i in temp1:
#                        print(TsAnalyzer.ts_VMD(i,k=k)[0].shape)
                    return np.array([TsAnalyzer.ts_VMD(i,k=k)[0]  for i in temp1])
#                return np.array([TsAnalyzer.ts_VMD(i,K=k)[0][:-pad_l,:]  for i in temp1])
        elif decompose == 'ewt':
            print(decompose)
            if pad_raw:  
                if pad_l:
                    pass
                else:    
                    
                    return np.array([np.hstack((i.reshape(-1,1),TsAnalyzer_.ts_EWT(i,k)[0]))  for i in temp1])
                
            else:
                if pad_l: 
                
                    return np.array([TsAnalyzer.ts_EWT(i,k)[0][:-pad_l,:]  for i in temp1])
                else:
                    return np.array([TsAnalyzer.ts_EWT(i,k)[0]  for i in temp1])


'''=====================================Temporal Analyzer of time series  ======================================
including: 
Signal processing tool:
DWT, SWT,      FFT ,      EMD,CEEMD,EEMD,    VMD,    EWT
Statistical tool:
Auto-correlation(ACF),       Cross-correlation(XCF),
'''
from vmdpy import VMD
import pywt
from PyEMD import EMD
import ewtpy

import matplotlib.pyplot as plt





class TsAnalyzer(object):
    def __init__(self):
        pass

    def ts_DWT(self,s, wavelet, mode,level):
        """
        :return: list [An, Dn, Dn-1, …, D2, D1]
        """
        coeffs = pywt.wavedec(s, wavelet, mode = mode, level=level)
        print(coeffs.shape)
        subseries = []
        for i in range(len(coeffs)):
            coeffs_i = [np.zeros_like(i) for i in coeffs]
            coeffs_i[i] = coeffs[i]
            s_i =  pywt.waverec(coeffs_i, wavelet, mode='symmetric')
            subseries.append(s_i)
        return subseries


    def ts_EMD(self,s,max_imf):
        """
        :param args:
        :return: k*len nd-array
        """
        emd = EMD()
        IMFs = emd(s, max_imf=max_imf)
        return IMFs

    def ts_EWT(self,s,k):
        """
        :param k: number of components
        :return: ewt--  len*k   nd array,mfb -filter bank, boundaries: frequency boundaries
        """
        ewt,  mfb ,boundaries = ewtpy.EWT1D(s, N = k)
        return ewt, mfb, boundaries

    def ts_VMD(self,x, display=0, alpha = 100,tau = 0.01,k = 4,DC = 0,init = 1, tol = 1e-5 ):
        """
        alpha = 2000       # moderate bandwidth constraint
        tau = 0.1           # noise-tolerance (no strict fidelity enforcement)
        K = 4              # 3 modes
        DC = 1             # no DC part imposed
        init = 1           # initialize omegas uniformly
        tol = 1e-7
        return: vmd_x   K*len   nd array
                vmd_spectrum  频谱
                每一个分量的中心频率
        """
        vmd_x, vmd_spectrum, centre_omega = VMD(x, alpha, tau, k, DC, init, tol)

        if display == 1:
            plt.figure()
            for i in range(1, 4):
                plt.subplot(3, 1, i)
                plt.plot(vmd_x[i - 1, :], c="b", linewidth=1)
                plt.title("Centre frequency is %.3f" % centre_omega[-1, i - 1])
            plt.suptitle("VMD of given series")
            plt.show()

        return vmd_x, vmd_spectrum, centre_omega


'''=====================================Benchmark forecasting models   ======================================
Statistical forecasting:
SARIMA + AIC 
ES:

Machine learning forecasting:
Linear regression,  SVM,  Gaussian Process,  Random forest,   LightGBM, XGboost

Neural Network forecasting:
MLP,  CNN,  LSTM 

Latest forecast model(TBC):
TCN, deepAR, deepState, Prophet


'''

import statsmodels.api as sm
import itertools
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
#import xgboost as xgb
import gc
from torch import nn
import torch.nn.functional as F


class TsBench(object):
    def __init__(self):
        pass

    def ts_SARIMA(train_ts, aid_ts, test_ts, step, search = False):
        """
        传入的 训练数据，然后拟合出pdq,spdq 然后然后 做k-step 预测，然后滑动一下，

        返回预测值, Param_aic 表

        """

        train_ts = train_ts
        aid_ts = aid_ts
        test_ts = test_ts
        k = step
        sw_ts = np.append(aid_ts, test_ts)[:-k]

        if search == True:
            # Define the p, d and q parameters to take any value between 0 and 2
            p = q = range(0, 3)
            d = [0,1]
            # Generate all different combinations of p, q and q triplets
            pdq = list(itertools.product(p, d, q))
            # Generate all different combinations of seasonal p, q and q triplets
            P =  Q = range(0, 3)
            D = [0,1]
            S = [0, 6, 12]
            PDQS = list(itertools.product(P, D, Q, S))

            best_aic = 1e6
            Param_aic_list = []
            best_pdq = (1,1,1)
            best_PDQS = (0,0,0,0)
            for param in pdq:
                for param_seasonal in PDQS:
                    if sum([i != 0 for i in param]) + sum([i != 0 for i in param_seasonal]) <= 6:
                        try:
                            mod = sm.tsa.statespace.SARIMAX(train_ts,
                                                            order=param,
                                                            seasonal_order=param_seasonal,
                                                            enforce_stationarity=False,
                                                            enforce_invertibility=False)

                            results = mod.fit(disp=0)

                            a = [i for i in param]
                            b = [i for i in param_seasonal]
                            c = [results.aic]
                            Params_aic = a + b + c

                            Param_aic_list.append(Params_aic)

                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_pdq = param
                                best_PDQS = param_seasonal

                        except:
                            continue

            train_mod = sm.tsa.statespace.SARIMAX(endog=train_ts,
                                                  order=best_pdq,
                                                  seasonal_order=best_PDQS,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
            train_results = train_mod.fit(disp=0)

        if search == False:
            best_pdq = (1,1,1)
            best_PDQS = (0,0,0,0)
            train_mod = sm.tsa.statespace.SARIMAX(endog=train_ts,
                                                  order=best_pdq,
                                                  seasonal_order=best_PDQS,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
            train_results = train_mod.fit(disp=0)

        forecast_SARIMA = []
        history = train_ts
        for i in range(len(sw_ts) + 1):
            test_mod = sm.tsa.SARIMAX(endog=history,
                                      order=best_pdq,
                                      seasonal_order=best_PDQS,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
            res = test_mod.filter(train_results.params)
            point_forecast = res.forecast(k)[-1]
            forecast_SARIMA.append(point_forecast)
            try:
                history = np.append(history, sw_ts[i])
            except:
                continue
        # plt.figure()
        # plt.plot(test_ts,'b')
        # plt.plot(np.array(forecast_SARIMA),'r')
        if search == True:
            return forecast_SARIMA, Param_aic_list
        else:
            return forecast_SARIMA

    def ts_ES(x_train, x_test, k):
        pass

    def ts_LR(X_train, y_train, X_test, search = False, L1 = 0.001):#with LR_1 penalty
        """
        a simple LR(LASSO) regression
        input: X_train, y_train, X_test, order=k, L1 penalty default = 0
        output: returns y_test
        """


        if search == False:
            model_LR = Lasso(alpha=L1, max_iter=10e5)
            model_LR.fit(X_train,y_train)
            y_hat = model_LR.predict(X_test)
            return y_hat

        if search == True:
            lvalid = len(X_test)
            X_valid = X_train[-lvalid:,]
            y_valid = y_train[-lvalid:]
            L1_list = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
            best_vloss = 1e6
            for L1_coef in L1_list:
                print('L1 coef = %.4f'%L1_list)
                model_LR = Lasso(alpha=L1_coef, max_iter=10e5)
                model_LR.fit(X_train[:-lvalid,],y_train[:-lvalid])
                y_hat = model_LR.predict(X_valid)
                vloss = np.linalg.norm(y_hat - y_valid,1)/len(y_hat)
                if vloss<best_vloss:
                    best_alpha = L1_coef
            best_model = Lasso(alpha=best_alpha, max_iter=10e5)
            best_model.fit(X_train,y_train)
            y_hat = best_model.predict(X_test)
            return y_hat

    def ts_SVM(X_train, y_train, X_test, search = False ):  #with LR_1 penalty

        if search == False:
            model_SVR = SVR(kernel='rbf')
            model_SVR.fit(X_train,y_train)
            y_hat = model_SVR.predict(X_test)
            return y_hat

        if search == True:
            lvalid = len(X_test)
            X_valid = X_train[-lvalid:,]
            y_valid = y_train[-lvalid:]
            kernel_list =  ['poly','rbf','sigmoid']
            C_list = [1, 10, 100, 1000] #loss的惩罚
            Gamma_list = [1e-5,1e-3,1e-1]
            HParams = list(itertools.product(kernel_list, C_list, Gamma_list))
            best_aic = 1e6
            best_vloss = 1e6
            for HP in HParams:
                model_SVR = SVR(kernel=HP[0],C=HP[1],gamma=HP[2])
                model_SVR.fit(X_train[:-lvalid,],y_train[:-lvalid])
                y_hat = model_SVR.predict(X_valid)
                vloss = np.linalg.norm(y_hat - y_valid,1)/len(y_hat)
                print(vloss)
                if vloss<best_vloss:
                    best_HP = HP
            best_model = SVR(kernel=best_HP[0],C=best_HP[1],gamma=best_HP[2])
            best_model.fit(X_train,y_train)
            y_hat = best_model.predict(X_test)
            return y_hat
        return

    def ts_GP(X_train, y_train, X_test, k):
        pass

    def ts_RF(X_train, y_train, X_test, k):
        pass

    

    def ts_LGBM(X_train, y_train, X_test, k):
        pass

    def ts_MLP(X_train, y_train, X_test, hidden_size, activation = 1, epoch=500, lr = 1e-2):
        """
        :hidden_size 隐藏层的神经元数量
        :activation:1 --relu; 2--tanh;3--sigmoid
        :return:
        """
        mean_X, std_X, norm_X = TsPrep.ts_normalize(X_train)
        mean_y, std_y, norm_y = TsPrep.ts_normalize(y_train)
        train_X = torch.from_numpy(norm_X).float()
        train_y = torch.from_numpy(norm_y).float()
        D_in = np.shape(X_train)[1]
        model =MLP(D_in ,hidden_size, 1,activation)
        loss_fn = nn.MSELoss(reduction = 'sum')
        optimizer = torch.optim.Adam(model.parameters(),lr = lr )
        #Loss = []
        for i in range(epoch):
            y_pred = model(train_X)
            loss= loss_fn(torch.squeeze(y_pred),train_y)
            #Loss.append(loss)
            if i%100==0:
                print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    #one-step

        Test_X = torch.from_numpy((X_test-mean_X)/std_X).float()
        norm_Test_y = model(Test_X).detach().numpy()
        Test_y = norm_Test_y*std_y+mean_y
        return Test_y


    def ts_LSTM(X_train, y_train, X_test, hidden_size, num_layers, epoch=300):
        """
        :param y_train:
        :param X_test:
        :param search:

        :return:
        """
        input_size = np.shape(X_train)[1]
        model = lstm_reg(input_size, hidden_size, output_size=1, num_layers=num_layers)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        mean_X, std_X, norm_X = TsPrep.ts_normalize(X_train,1)
        mean_y, std_y, norm_y = TsPrep.ts_normalize(y_train,1)

        train_X = norm_X.reshape(-1, 1, input_size)
        train_y = norm_y.reshape(-1, 1, 1)

        train_X = torch.from_numpy(train_X).float()
        train_y = torch.from_numpy(train_y).float()

        # train_X = torch.tensor(train_X, dtype = torch.float32)
        # train_y = torch.tensor(train_y, dtype = torch.float32)

        for e in range(epoch):
            out = model(train_X)
            last_hidden = model.last_hidden(train_X)

            loss = loss_function(out, train_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (e + 1) % 20 == 0: # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

        model.eval()
        norm_test_X = (X_test - mean_X)/std_X
        test_X = norm_test_X.reshape(-1, 1, input_size)
        test_X = torch.from_numpy(test_X).float()
        lstm_hat = model(test_X,last_hidden)
        lstm_hat = lstm_hat.view(-1).data.numpy()*std_y + mean_y

        return lstm_hat

    def ts_CNN(X_train, y_train, X_test,epoch= 500):
        """


        """
        order = np.shape(X_train)[1]
        fc_neurons = int(np.floor((np.floor(order/2)-1)/2-1))*16
        if order <12:
            fc_neurons = int(np.floor(order/2)-1)*8

        mean_X, std_X, X_train =  TsPrep.ts_normalize(X_train,1)
        mean_y, std_y, y_train =  TsPrep.ts_normalize(y_train,1)
        X_test = (X_test - mean_X)/std_X


        X_train = torch.from_numpy(X_train).float()  #batch size , in_channel = 1 , length = 50
        X_train = X_train.unsqueeze(1)
        # look_back = 12 /24/36
        y_train = torch.from_numpy(y_train).float()
        y_train = y_train.unsqueeze(1)

        X_test = torch.from_numpy(X_test).float()  #batch size , in_channel = 1 , length = 50
        X_test = X_test.unsqueeze(1)

        lr = 1e-2
        model = CNN_pred(fc_neurons, order)
        loss_fn = nn.MSELoss(reduction = 'sum')
        optimizer = torch.optim.Adam(model.parameters(),lr = lr )
        #Loss = []
        for i in range(epoch):
            y_pred = model(X_train)
            loss= loss_fn(y_train,y_pred)
            #Loss.append(loss)
            if i%100==0:
                print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    #one-step

        out_y = model(X_test).detach().squeeze().data.numpy()

        return out_y*std_y + mean_y





    def ts_TCN(X_train, y_train, X_test, k):
        pass

    def ts_DeepAR(X_train, y_train, X_test, k):
        pass

    def ts_DeepSSM(X_train, y_train, X_test, k):
        pass

    def tS_Prophet(X_train, y_train, X_test, k):
        pass


'''===================================== 评估预测结果   ======================================
Statistical :
QQplot, Residual plot,  白噪声检验, Pearson相关系数, AIC/BIC

Machine learning:
RMSE, MASE, MASE, MAPE, Hit rate ,sMAPE
    

'''


class TsMetric(object):
    def __init__(self):
        pass
    def RMSE(self,actual, pred):
        """
        RMSE = sqrt(1/n * sum_{i=1}^{n}{pred_i - actual_i} )
        input: actual and pred should be np.array
        output: RMSE
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        RMSE = np.sqrt( 1/len(actual) *np.linalg.norm(actual - pred,2)**2)
        return RMSE
    def MBE(self,actual, pred):
        '''
        MAE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |
        input: actual and pred should be np.array
        output: MAE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MBE =  1/len(actual) *np.sum(pred-actual)
        return MBE

    def MAE(self,actual, pred):
        '''
        MAE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |
        input: actual and pred should be np.array
        output: MAE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        return MAE


    def MASE(self,actual, pred, history):
        '''
        MASE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |/ sum_traning(|diff|)
        input: actual and pred should be np.array
        output: MASE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        Scale =  1/(len(history)-1) * np.linalg.norm(np.diff(history),1)
        MASE = MAE/Scale

        return MASE

    def MAPE(self,actual, pred):
        '''
        MAPE = 1/n * sum_{i=1}^{n} |pred_i - actual_i} |/|actual_i|
        input: actual and pred should be np.array
        output: MAPE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAPE =  1/len(actual) *np.linalg.norm((actual - pred)/actual, 1)

        return MAPE

    def sMAPE(actual, pred):
        """
        1/n  *  SUM_{i=1 to n}  { ( |pred_i-actual_i|)   /  (0.5*|pred_i|+0.5*|actual_i|)}
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        sMAPE = 1/len(actual) * np.sum(2*np.abs(actual - pred)/(np.abs(actual)+np.abs(pred)))
        return sMAPE

    def RAE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with\
        :return:
            l1(pred-actual)/ l1(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,1)
        denom = np.linalg.norm(actual - compared,1)
        return nom/denom

    def RSE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with
        :return:
            l2(pred-actual)/ l2(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,2) ** 2
        denom = np.linalg.norm(actual - compared,2) ** 2
        return nom/denom

    def Corr(actual, pred):
        """
        :param actual: 
        :param pred: 
        return     np.dot(actual - mean(actual), pred -mean(pred)) / norm(actual-mean(acutal)*norm(pred- mean(pred))
        """
        nom = np.dot(actual -np.mean(actual), pred-np.mean(pred))
        denom =np.linalg.norm(actual-np.mean(actual)) * np.linalg.norm(pred- np.mean(pred))
        return nom/denom


'''========================上面用到的一些函数和调参==================================='''
"""
1. XGB的贝叶斯调参
2. lstm的类 定义和贝叶斯调参

"""
from hyperopt import hp,tpe,partial,fmin


def Bayes_param_tuning(seed, num):

        #定义域空间

    """
    choice：类别变量
    quniform：离散均匀（整数间隔均匀）
    uniform：连续均匀（间隔为一个浮点数）
    loguniform：连续对数均匀（对数下均匀分布）
    """

    space = {
        'n_estimators':  hp.quniform('n_estimators', 200, 2000, 100),
    #    'booster': 'gbtree',
    #   'objective': 'reg:linear',
    #   'eval_metric': 'mae',
        'gamma': hp.loguniform('gamma', np.log(0.01), np.log(0.2)),
        'max_depth': hp.quniform('max_depth', 5, 8, 1),
        'lambda': hp.uniform('lambda', 0.0, 1),
        'alpha': hp.uniform('alpha', 0.0, 1),
        'subsample': hp.uniform('subsample', 0.7, 1),
        'colsample_bytree':hp.uniform('colsample_bytree', 0.7, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
   #     'silent': 1,
        'eta': hp.loguniform('eta', np.log(0.01), np.log(0.2))
   #     'seed': 1000
        }

    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(XGboost_eval_MAE,space, algo=algo, max_evals=num, verbose = True
                ,rstate= np.random.RandomState(seed))
    return best



def XGboost_eval_MAE(argsDict):
    params = {
            'n_estimators':argsDict['n_estimators'],
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'gamma': argsDict['gamma'],
            'max_depth': int(argsDict['max_depth']),
            'lambda': argsDict['lambda'],
            'lambda': argsDict['alpha'],
            'subsample': argsDict['subsample'],
            'colsample_bytree': argsDict['colsample_bytree'],
            'min_child_weight': argsDict['min_child_weight'],
            'silent': 1,
            'eta': argsDict['eta'],
            'seed': 1000
              }

    dtrain_temp = xgb.DMatrix(X_train_xgb, y_train_xgb)
    dvalid_temp = xgb.DMatrix(X_valid_xgb, y_valid_xgb)
    watchlist = [(dtrain_temp, 'train'), (dvalid_temp, 'valid')]
    num_rounds = 1000

    xrf = xgb.train(params, dtrain_temp, num_rounds,evals = watchlist,
                    verbose_eval=False, early_stopping_rounds=50)
    MAE = np.linalg.norm(y_valid_xgb - xrf.predict(xgb.DMatrix(X_valid_xgb)),1)
    return MAE


class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size=1,num_layers=3):
        super(lstm_reg,self).__init__()

        self.rnn = nn.LSTM(input_size,hidden_size,num_layers)  #hidden size 是隐藏层神经元个数 也是每个门输出的维度， num_layer 是 lstm 堆叠的层数 ；input(seq_len, batch, input_size)
        self.reg = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden=0):
        if hidden is not 0:
            x, _ = self.rnn(x,hidden)
        else:
            x, _ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x

    def last_hidden(self,x):
        _, hidden = self.rnn(x)
        return hidden


class MLP(nn.Module):
    def __init__(self,D_in ,H, D_out, activation):   #framework definition
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H, bias=True)
        self.linear2 = torch.nn.Linear(H,D_out, bias=True)
        self.activation = activation
    def forward(self,x):                #forward computing

        if self.activation ==1:
            y_pred = self.linear2(F.relu(self.linear1(x)))
        elif self.activation ==2:
            y_pred = self.linear2(torch.tanh(self.linear1(x)))
        else:
            y_pred = self.linear2(torch.sigmoid(self.linear1(x)))
        return y_pred


class CNN_pred(nn.Module):
    def __init__(self, fc_nodes, order):
        super(CNN_pred,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = 8,
                kernel_size = 3
            ),
            nn.MaxPool1d(
                kernel_size =2
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 3
            ),
            nn.MaxPool1d(
                kernel_size =2
            )
        )
        self.fc = nn.Linear(fc_nodes,1)  #需要计算  #512  16 1
        self.order = order

    def forward(self, X):
        y = self.conv1(X)
        if self.order>=12:
            y = self.conv2(y)
        y = y.view(y.size(0),-1)
        out = self.fc(y)
        return out

