# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:06:40 2020

@author: lenovo
"""
from sklearn import preprocessing
from ForecastLib import TsMetric
import numpy as np
import pandas as pd
import numpy.ma as ma
import collections
import matplotlib.pyplot as plt
from scipy.stats import rankdata,wilcoxon,friedmanchisquare
#ewtmedianp[:,k]
import scikit_posthocs as sp # https://pypi.org/project/scikit-posthocs/
#import stac
# Helper functions for performing the statistical tests
def generate_scores(method, method_args, data, labels):
    pairwise_scores = method(data, **method_args) # Matrix for all pairwise comaprisons
    pairwise_scores.set_axis(labels, axis='columns', inplace=True) # Label the cols
    pairwise_scores.set_axis(labels, axis='rows', inplace=True) # Label the rows, note: same label as pairwise combinations
    return pairwise_scores
def remove_minmax(x):
    xp=x.copy()
    max_=x.max()
    min_=x.min()
    # print(xp.argmax())
    xp=np.delete(xp,xp.argmax())
    xp=np.delete(xp,xp.argmin())
    return xp
def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd
def get_prediction(name):
    #file_name = D:\Newbuilding price forecasting\Corrected results
    file_name = 'D:\\Newbuilding price forecasting (Word)\\Corrected results\\'+name+'.csv'
  
    dat = pd.read_csv(file_name)
    # aa=np.round(dat.values,2)
    dat = dat.fillna(method='ffill')
    # dat.values=aa
    return dat,dat.columns
def get_data(name):
    #file_name = 'C:\\Users\\lenovo\\Desktop\\FuzzyTimeSeries\\pyFTS-master\\pyFTS\\'+name+'.csv'
    file_name = name+'.csv'
    #D:\Multivarate paper program\monthly_data
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    # if 'AEMO' not in name:
    #     dat.values=np.round(dat.values,2)
    return dat#,dat.columns
def compute_pre(x):
    #x(nsample, nl*nseed)
    nsample=x.shape[0]
    offset=10-nl
    # print(nl)
    seed=10
    each_seed=np.zeros((nsample,10))
    for i in range(seed):
        cp=x[:,i*10:i*10+nl]
        each_seed[:,i]=np.mean(cp,axis=1)
    return each_seed
# def compute_pre2(x):
#     #x(nsample, nl*nseed)
#     nsample=x.shape[0]
#     nl=12
#     seed=10
#     each_seed=np.zeros((nsample,10))
#     for i in range(seed):
#         cp=x[:,i*nl:(i+1)*nl-2]
#         each_seed[:,i]=np.mean(cp,axis=1)
#     return each_seed
# loc='SA'
# year='2020'
# month='PRICE_AND_DEMAND_202010_'+loc+str(1)#'PRICE_AND_DEMAND_202001_QLD1'
# dataset='D:\\AEMO\\'+loc+'\\'+ year +'\\'+month

rmse_all=[]
mape_all=[]
mase_all=[]


mss=['Jan','Apr','Jul','Oct']
year='2020'
locs=['SA','NSW','VIC','TAS']
months=['01_','04_','07_','10_']
NLS=[2,4,6,8,10]
# labels=['MASE']#['RMSE','MASE','MAPE']
BOAedRVFL_means=[]
GridEWTedRVFL_means=[]
BOAEWTedRVFL_means=[]

BOAedRVFL_rmsemeans=[]
GridEWTedRVFL_rmsemeans=[]
BOAEWTedRVFL_rmsemeans=[]
for nl in NLS:
    for loc in locs:
        axi=0
        # fig, axes = plt.subplots(4,1,figsize=(9,9))
        plt.rcParams["font.family"] = "Times New Roman"
    
        for month_ in months:
            month='PRICE_AND_DEMAND_2020'+month_+loc+str(1)
            dataset='D:\\AEMO\\'+loc+'\\'+ year +'\\'+month
    #                print(month)
            pddata=get_data(dataset)
            data_=pddata['TOTALDEMAND'].values.reshape(-1,1)
           
    # for co in countrys[:]:
        # loc+='.txt'
        # axi=0
        # fig, axes = plt.subplots(4,1,figsize=(9,9))
        # plt.rcParams["font.family"] = "Times New Roman"
        # loc=co
        # for month in [1,4,7,10]:
    
            # month='PRICE_AND_DEMAND_2020'+month+loc+str(1)#'PRICE_AND_DEMAND_202001_QLD1'
            # c_data=df_data.loc[(df_data['Country']==co)&(df_data['Year']==year)&(df_data['Month']==month)]
            # data_=c_data[hours].values.reshape(-1,1)
            np_data=data_#df_data.loc[df_data['MM']==month]['WSPD'].values.astype(float).reshape(-1,1)
            # scaler=preprocessing.MinMaxScaler()
            validation_l,test_l=int(0.1*np_data.shape[0]),int(0.2*np_data.shape[0])
            train_l=len(np_data)-test_l-validation_l
            target=np_data[-test_l:].ravel()
            history=np_data[:-test_l].ravel()
            prediction={}
            pre_loc='D:\\Walkforwarddecomposition\\Results\\'
            rmse={}
            mape={}
            mase={}
            error={}
            
            metric=TsMetric()
            
            # #MAPE
            # #MASE
            #BOA+EDRVFL
            boaedrvfl_loc='AEMO/ABA/allpedRVFLBOA1250'+month+'.csv'
            boaedrvflpres=get_data(boaedrvfl_loc).values[:,1:]#297*100
            boaedrvflpres=compute_pre(boaedrvflpres)
            prediction['edRVFLBOA']=boaedrvflpres#.mean(axis=1)
            boaedrvfl_rmse=np.zeros(boaedrvflpres.shape[1])
            boaedrvfl_mape=np.zeros(boaedrvflpres.shape[1])
            boaedrvfl_mase=np.zeros(boaedrvflpres.shape[1])
            for k in range(boaedrvflpres.shape[1]):
                boaedrvfl_rmse[k]=metric.RMSE(target, boaedrvflpres[:,k])
                boaedrvfl_mape[k]=metric.MAPE(target, boaedrvflpres[:,k])
                boaedrvfl_mase[k]=metric.MASE(target, boaedrvflpres[:,k],history)
            rmse['edRVFLBOA']=boaedrvfl_rmse.mean()
            mase['edRVFLBOA']=boaedrvfl_mase.mean()
            mape['edRVFLBOA']=boaedrvfl_mape.mean()
            
            
           
            
            gridewtedrvfl_loc='AEMO/ABA/allpEWTedRVFLGrid1250'+month+'.csv'
            gridewtedrvflpres=get_data(gridewtedrvfl_loc).values[:,1:]#297*100
            gridewtedrvflpres=compute_pre(gridewtedrvflpres)
            prediction['EWTedRVFLGrid']=gridewtedrvflpres#.mean(axis=1)
            gridewtedrvfl_rmse=np.zeros(gridewtedrvflpres.shape[1])
            gridewtedrvfl_mape=np.zeros(gridewtedrvflpres.shape[1])
            gridewtedrvfl_mase=np.zeros(gridewtedrvflpres.shape[1])
            for k in range(gridewtedrvflpres.shape[1]):
                gridewtedrvfl_rmse[k]=metric.RMSE(target, gridewtedrvflpres[:,k])
                gridewtedrvfl_mape[k]=metric.MAPE(target, gridewtedrvflpres[:,k])
                gridewtedrvfl_mase[k]=metric.MASE(target, gridewtedrvflpres[:,k],history)
            rmse['EWTedRVFLGrid']=gridewtedrvfl_rmse.mean()
            mase['EWTedRVFLGrid']=gridewtedrvfl_mase.mean()
            mape['EWTedRVFLGrid']=gridewtedrvfl_mape.mean()
            #allpEWTedRVFLBOA
            boaewtedrvfl_loc='AEMO/ABA/allpEWTedRVFLBOA1250'+month+'.csv'
            boaewtedrvflpres=get_data(boaewtedrvfl_loc).values[:,1:]#297*100
            boaewtedrvflpres=compute_pre(boaewtedrvflpres)
            prediction['EWTedRVFLBOA']=boaewtedrvflpres#.mean(axis=1)
            boaewtedrvfl_rmse=np.zeros(boaewtedrvflpres.shape[1])
            boaewtedrvfl_mape=np.zeros(boaewtedrvflpres.shape[1])
            boaewtedrvfl_mase=np.zeros(boaewtedrvflpres.shape[1])
            for k in range(boaewtedrvflpres.shape[1]):
                boaewtedrvfl_rmse[k]=metric.RMSE(target, boaewtedrvflpres[:,k])
                boaewtedrvfl_mape[k]=metric.MAPE(target, boaewtedrvflpres[:,k])
                boaewtedrvfl_mase[k]=metric.MASE(target, boaewtedrvflpres[:,k],history)
            rmse['EWTedRVFLBOA']=boaewtedrvfl_rmse.mean()
            mase['EWTedRVFLBOA']=boaewtedrvfl_mase.mean()
            mape['EWTedRVFLBOA']=boaewtedrvfl_mape.mean()
           
            
            # edrvfl_dt_loc='D:\\DeepESN-master\\AEMO\\dESNBOA'+loc+str(month_)#dESNBOA
            # edrvfl_dtpres=get_data(edrvfl_dt_loc).values[:,1:]
            # prediction['edRVFLDT']=edrvfl_dtpres.mean(axis=1)
            # edrvfl_dt_rmse=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mape=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mase=np.zeros(edrvfl_dtpres.shape[1])
            # for k in range(edrvfl_dtpres.shape[1]):
            #     edrvfl_dt_rmse[k]=metric.RMSE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mape[k]=metric.MAPE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mase[k]=metric.MASE(target, edrvfl_dtpres[:,k],history)
            # rmse['dESNBOA']=edrvfl_dt_rmse.mean()
            # mase['dESNBOA']=edrvfl_dt_mase.mean()
            # mape['dESNBOA']=edrvfl_dt_mape.mean()
            
            # edrvfl_dt_loc='D:\\DeepESN-master\\AEMO\\EWTRVFLBOA50'+str(month)#dESNBOA
            # edrvfl_dtpres=get_data(edrvfl_dt_loc).values[:,1:]
            # prediction['edRVFLDT']=edrvfl_dtpres.mean(axis=1)
            # edrvfl_dt_rmse=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mape=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mase=np.zeros(edrvfl_dtpres.shape[1])
            # for k in range(edrvfl_dtpres.shape[1]):
            #     edrvfl_dt_rmse[k]=metric.RMSE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mape[k]=metric.MAPE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mase[k]=metric.MASE(target, edrvfl_dtpres[:,k],history)
            # rmse['EWTRVFL']=edrvfl_dt_rmse.mean()
            # mase['EWTRVFL']=edrvfl_dt_mase.mean()
            # mape['EWTRVFL']=edrvfl_dt_mape.mean()
            
            # edrvfl_dt_loc='D:\\DeepESN-master\\AEMO\\edRVFLBOA50'+str(month)#dESNBOA
            # edrvfl_dtpres=get_data(edrvfl_dt_loc).values[:,1:]
            # prediction['edRVFLDT']=edrvfl_dtpres.mean(axis=1)
            # edrvfl_dt_rmse=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mape=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mase=np.zeros(edrvfl_dtpres.shape[1])
            # for k in range(edrvfl_dtpres.shape[1]):
            #     edrvfl_dt_rmse[k]=metric.RMSE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mape[k]=metric.MAPE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mase[k]=metric.MASE(target, edrvfl_dtpres[:,k],history)
            # rmse['edRVFL']=edrvfl_dt_rmse.mean()
            # mase['edRVFL']=edrvfl_dt_mase.mean()
            # mape['edRVFL']=edrvfl_dt_mape.mean()
            
            
            # edrvfl_dt_loc='D:\\DeepESN-master\\AEMO\\EWTedRVFLBOA50'+str(month)
            # edrvfl_dtpres=get_data(edrvfl_dt_loc).values[:,1:]
            # prediction['RVFL']=edrvfl_dtpres.mean(axis=1)
            # edrvfl_dt_rmse=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mape=np.zeros(edrvfl_dtpres.shape[1])
            # edrvfl_dt_mase=np.zeros(edrvfl_dtpres.shape[1])
            # for k in range(edrvfl_dtpres.shape[1]):
            #     edrvfl_dt_rmse[k]=metric.RMSE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mape[k]=metric.MAPE(target, edrvfl_dtpres[:,k])
            #     edrvfl_dt_mase[k]=metric.MASE(target, edrvfl_dtpres[:,k],history)
            # rmse['EWTedRVFLBOA']=edrvfl_dt_rmse.mean()
            # mase['EWTedRVFLBOA']=edrvfl_dt_mase.mean()
            # mape['EWTedRVFLBOA']=edrvfl_dt_mape.mean()
            error['RMSE']=rmse
            error['MAPE']=mape
            error['MASE']=mase
            error_df=pd.DataFrame.from_dict(error,orient='index')
            # e_df=error_df.reindex(['RMSE','MAPE','MASE'])
            rmse_all.append(error_df.loc['RMSE'].values.reshape(1,-1))
            mape_all.append(error_df.loc['MAPE'].values.reshape(1,-1))
            mase_all.append(error_df.loc['MASE'].values.reshape(1,-1))
            
            # print(e_df)
        # fig.show()
        # fig.savefig(pre_loc+'EWTedRVFLBOA50'+loc+'.eps', dpi=1000,format='eps')
    rmse_all_np=np.concatenate(rmse_all,axis=0)
    mase_all_np=np.concatenate(mase_all,axis=0)
    mape_all_np=np.concatenate(mape_all,axis=0)
    scaler=preprocessing.MinMaxScaler()
    rmse_all_np=scaler.fit_transform(rmse_all_np.T).T
    mase_all_np=scaler.fit_transform(mase_all_np.T).T
    # rmse_all_np=rmse_all_np/rmse_all_np.max(axis=1)[:,None]
    # mase_all_np=mase_all_np/mase_all_np.max(axis=1)[:,None]
    rmsealldf=pd.DataFrame(data=rmse_all_np,columns=error_df.columns)
    masealldf=pd.DataFrame(data=mase_all_np,columns=error_df.columns)
    mapealldf=pd.DataFrame(data=mape_all_np,columns=error_df.columns)
    
    ranks=np.zeros(rmse_all_np.shape)
    av=[]
    for err in [rmse_all_np,mase_all_np,mape_all_np]:
        for i in range(err.shape[0]):
            #     print(err.values[:,1:][i,:])
            ranks[i,:]=rankdata(err[i,:])
        # af1=friedmanchisquare(err[:,0],err[:,1],err[:,2],err[:,3],err[:,4],err[:,5],
        #                   err[:,6],err[:,7],err[:,8],err[:,9],err[:,10],err[:,11],
        #                   err[:,12],err[:,13],err[:,14])
        af2=friedmanchisquare(*err)
        print(af2)
        avranks=np.mean(ranks,axis=0).reshape(1,-1)
        cd=compute_CD(avranks.ravel(),err.shape[0])
        av.append(avranks)
        # print(avranks)
    avrank=np.concatenate(av,axis=0)
    avrank_df=pd.DataFrame(data=avrank,columns=error_df.columns)
    print(avrank_df)
    # labels=['MASE']#['RMSE','MASE','MAPE']
    BOAedRVFL_means.append(masealldf['edRVFLBOA'].values.mean())
    GridEWTedRVFL_means.append(masealldf['EWTedRVFLGrid'].values.mean())
               
    BOAEWTedRVFL_means.append(masealldf['EWTedRVFLBOA'].values.mean())
    BOAedRVFL_rmsemeans.append(rmsealldf['edRVFLBOA'].values.mean())
    GridEWTedRVFL_rmsemeans.append(rmsealldf['EWTedRVFLGrid'].values.mean())
               
    BOAEWTedRVFL_rmsemeans.append(rmsealldf['EWTedRVFLBOA'].values.mean())
x = np.arange(len(NLS))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, BOAedRVFL_means, width, label='BOAedRVFL')
rects2 = ax.bar(x ,  GridEWTedRVFL_means, width, label='GridEWTedRVFL')
rects3 = ax.bar(x + width/2, BOAEWTedRVFL_means, width, label='BOAEWTedRVFL')

# Add some text for labels, title and custom x-axis tick labels, etc.
labels=NLS
ax.set_ylabel('Errors')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, BOAedRVFL_rmsemeans, width, label='BOAedRVFL')
rects2 = ax.bar(x ,  GridEWTedRVFL_rmsemeans, width, label='GridEWTedRVFL')
rects3 = ax.bar(x + width, BOAEWTedRVFL_rmsemeans, width, label='BOAEWTedRVFL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized RMSE')
ax.set_xlabel('Layers')
# ax.set_title('Scores by group and gender')
ax.set_xticks( x)
ax.set_xticklabels(labels)
# ax.legend()
ax.legend(loc='center left', framealpha=0,fontsize=12,bbox_to_anchor=(1, 0.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.legend(framealpha=0,fontsize=7)
ax.set_facecolor('white')
fig.tight_layout()
plt.show()
plt.savefig('ablationstudyRMSE.jpg',dpi=1000,format='jpg', bbox_inches = 'tight' ,  pad_inches = 0)
plt.savefig('ablationstudyRMSE.eps',dpi=1000,format='eps', bbox_inches = 'tight' ,  pad_inches = 0)
# #cd 4.796
# # rank(*err)
# rmse_nemenyi_scores = generate_scores(sp.posthoc_nemenyi_friedman, {}, rmsealldf.values, avrank_df.columns)
# mase_nemenyi_scores = generate_scores(sp.posthoc_nemenyi_friedman, {}, masealldf.values, avrank_df.columns)
# mape_nemenyi_scores = generate_scores(sp.posthoc_nemenyi_friedman, {}, mapealldf.values, avrank_df.columns)