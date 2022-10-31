'''
EWT edRVFL BOA

----
'''
from sklearn.tree import DecisionTreeRegressor
from itertools import product
import matplotlib.pyplot as plt
import ForecastLib
from sklearn import preprocessing
import numpy as np
import random
import pandas as pd
from hyperopt import fmin, tpe, hp
# import DeepRVFL_
from DeepRVFL_.DeepRVFL import DeepRVFL
# import DeepRVFL
from utils import MSE, config_MG, load_MG
from skfeature.function.similarity_based import reliefF,lap_score
def feature_importance(x,y,method='DT'):
    scaler=preprocessing.MinMaxScaler()
    x=scaler.fit_transform(x)
    if method=='DT':
        model=DecisionTreeRegressor(criterion='mse')
        model.fit(x,y)
        importance=model.feature_importances_
    elif method=='LR':
        model=DecisionTreeRegressor(criterion='mse')
        model.fit(x,y)
        importance=model.feature_importances_
    return importance
def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]
def format_data(dat,order,idx=0):
    n_sample=dat.shape[0]-order
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =dat[i+order,idx]
    return x.T,y.T
def select_indexes(data, indexes):

    # if len(data) == 1:
    return data[:,indexes]
    
    # return [data[i] for i in indexes]
def compute_error(actuals,predictions,history=None):
    actuals=actuals.ravel()
    predictions=predictions.ravel()
    
    metric=ForecastLib.TsMetric()
    error={}
    error['RMSE']=metric.RMSE(actuals, predictions)
    # error['MAPE']=metric.MAPE(actuals,predictions)
    error['MAE']=metric.MAE(actuals,predictions)
    if history is not None:
        history=history.ravel()
        error['MASE']=metric.MASE(actuals,predictions,history)
        
    
    return error
def get_data(name):
    #file_name = 'C:\\Users\\lenovo\\Desktop\\FuzzyTimeSeries\\pyFTS-master\\pyFTS\\'+name+'.csv'
    file_name = name+'.csv'
    #D:\Multivarate paper program\monthly_data
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns
class Struct(object): pass

# sistemare indici per IP in config_pianomidi, mettere da un'altra parte
# sistema selezione indici con transiente messi all'interno della rete

def config_load(iss,IP_indexes):

    configs = Struct()
    
    
    configs.iss = iss # set insput scale 0.1 for all recurrent layers
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 # activate pre-train
    

#    configs.IPconf.Nepochs=10
    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 # connectivity of recurrent matrix
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge' # train with singular value decomposition (more accurate)
    # configs.readout.regularizations = 10.0**np.array(range(-16,-1,1))
    
    return configs 
def dRVFL_predict(hyper,norm_data,train_idx,test_idx,layer,s,last_states=None,dec_=None,alldecs=None):
   
    np.random.seed(s)
    w,k=hyper[0][3],hyper[0][4]
    data=Struct()
    if dec_ is None:
        dec_=alldecs[str(w)+str(k)]#decomposer.ts_walkforward(norm_data.ravel(),w,k,'ewt',pad_l=0,pad_raw=True)
    
   
    # print(dec_.shape,k)
    data.inputs=np.zeros((k*48+48,dec_.shape[0]))
    data.targets=np.zeros((1,dec_.shape[0]))
    for i in range(dec_.shape[0]):
        data.inputs[:,i]=dec_[i,-48:,:].ravel()
        data.targets[0,i]=norm_data[w+i]
    Nu=data.inputs.shape[0]
    val_l,test_l=int(0.1*norm_data.shape[0]),int(0.2*norm_data.shape[0])
    train_l=data.inputs.shape[1]-val_l-test_l
    # train_idx=range(train_l)
    # val_idx=range(train_l,train_l+val_l)
    # test_idx=range(train_l+val_l,data.inputs.shape[1])
    Nh = hyper[0][0] 
    # print(hyper[0][3])
    # ratio=hyper[0][3]
    Nl = layer # 
    
    reg=[]

    iss=[]
    for h in hyper:
        reg.append( h[1])        
        iss.append(h[2])
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    train_targets = select_indexes(data.targets, train_idx)
    # feature_score=reliefF.reliefF(data.inputs.T[:len(train_idx),:],train_targets.ravel())          
    # ranks=feature_ranking(feature_score)
    # idx1=ranks[:int(ratio*Nh)]
    if Nl==1:
        
        states = deepRVFL.computeLayerState(0,data.inputs)
    else:
        
        # print('max',max(importance),'sel',importance[idx[0]])
        states=deepRVFL.computeLayerState(Nl-1,data.inputs,last_states[:,:])
    # importance=feature_importance(states[:,:len(train_idx)].T, train_targets.T,method='DT')
    # idx=feature_ranking(importance)[:int(ratio*Nh)]      
    # states=states[idx,:]
    train_states = select_indexes(np.concatenate([states,data.inputs],axis=0), train_idx)#(Nh,n_sample)
    # print(train_states.shape,train_targets.shape)
    # importance=feature_importance(train_states.T, train_targets.T,method='DT')
    # idx=feature_ranking(importance)[:int(ratio*Nh)]
    test_states = select_indexes(np.concatenate([states,data.inputs],axis=0), test_idx)
    # feature_score=reliefF.reliefF(train_states.T,train_targets.ravel())          
    # ranks=feature_ranking(feature_score)
    # idx=ranks[:int(ratio*Nh)]
    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
 
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm,states[:,:],dec_
def edRVFL_predict(hyper,data,train_idx,test_idx,s):
    #idxs:list(train_idx) 
#    Nrs,Nls,regs,transients,spectral_radiuss,leaky_rates,input_scale
    np.random.seed(s)
    #[Nl,hidden,reg,iss,w,k]
    Nu=data.inputs.shape[0]
    Nr = hyper[1] # number of recurrent units
    # ratio=hyper[0][3]
    Nl = hyper[0]# number of recurrent layers
    reg=[hyper[2] for i in range(Nl)]
    iss=[hyper[3] for i in range(Nl)] 
    
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nr, Nl, configs)
    last_states=None
    outputs=np.zeros((len(test_idx),Nl))
    train_targets = select_indexes(data.targets, train_idx)
    for l in range(Nl):
        if l==0:
            # feature_score=reliefF.reliefF(data.inputs.T[:len(train_idx),:],train_targets.ravel())          
            # ranks=feature_ranking(feature_score)
            # idx1=ranks[:int(ratio*Nr)]
            states = deepRVFL.computeLayerState(l,data.inputs,inistate=None)
            # states = deepRVFL.computeGlobalState(data.inputs)
        else:
            
            # print(l,ratio)
            # importance=feature_importance(last_states[:,:len(train_idx)].T, train_targets.T,method='DT')
           
            # idx=feature_ranking(importance)[:int(ratio*Nr)]
            states=deepRVFL.computeLayerState(l,data.inputs,last_states)
        # ratio=hyper[l][3]
        # importance=feature_importance(states[:,:len(train_idx)].T, train_targets.T,method='DT')
        # idx=feature_ranking(importance)[:int(ratio*Nr)]      
        # states=states[idx,:]
        last_states=states
        train_states = select_indexes(np.concatenate([states,data.inputs],axis=0), train_idx)
        train_targets = select_indexes(data.targets, train_idx)
        # importance=feature_importance(train_states.T, train_targets.T,method='DT')
        # idx=feature_ranking(importance)[:int(ratio*Nr)]
        # feature_score=reliefF.reliefF(train_states.T,train_targets.ravel())          
        # ranks=feature_ranking(feature_score)
        # idx=ranks[:int(ratio*Nr)]
        # print(l,Nu,Nr)
        
        test_states = select_indexes(np.concatenate([states,data.inputs],axis=0), test_idx)
        # print(train_states.shape,Nr)
        deepRVFL.trainReadout(train_states, train_targets, reg[l])
        test_outputs_norm = deepRVFL.computeOutput(test_states).T
        outputs[:,l:l+1]=test_outputs_norm

    return np.mean(outputs,axis=1).reshape(-1,1),outputs#outputs.mean(axis=1).reshape(-1,1)
def RVFL_predict(hyper,data,train_idx,test_idx,s):
    #idxs:list(train_idx) 
#    Nrs,Nls,regs,transients,spectral_radiuss,leaky_rates,input_scale
    np.random.seed(s)
    Nu=1
    Nh = hyper[0] # number of recurrent units
    Nl = 1#hyper[1] # number of recurrent layers
    reg = hyper[1]
    iss=hyper[2]
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    states = deepRVFL.computeState(data.inputs, deepRVFL.IPconf.DeepIP)  
#    print(transient,states[0].shape,states[0][0,:2],states[0][0,-2:])              
    train_states = select_indexes(states, train_idx)
    train_targets = select_indexes(data.targets, train_idx)
    test_states = select_indexes(states, test_idx)
    outputs=np.zeros((len(test_idx),Nl))
    for i in range(Nl):              
        deepRVFL.trainReadout(train_states[i*Nh:i*Nh+Nh,:], train_targets, reg)
        test_outputs_norm = deepRVFL.computeOutput(test_states[i*Nh:i*Nh+Nh,:]).T
        outputs[:,i:i+1]=test_outputs_norm
#    test_outputs=scaler.inverse_transform(test_outputs_norm)
#    actuals=data_[-len(test_idx):]
#    test_err=compute_error(actuals,test_outputs,None)
    return np.mean(outputs,axis=1).reshape(-1,1)#outputs.mean(axis=1).reshape(-1,1)
def cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,regs,input_scale,ratios=[1],scaler=None,s=0,boat=50):
    best_hypers=[]
    np.random.seed(s)
    layer_s=None
    dec_s=None
    alldecs={}
    
    print('collect all dec series')
    loss=np.inf
    for w in [48,96,144]:
        for k in [2,3,4]:
            print(w,k)
            dec_=decomposer.ts_walkforward(data.ravel(),w,k,'ewt',pad_l=0,pad_raw=True)
            alldecs[str(w)+str(k)]=dec_
            dat=Struct()
            # print(dec_.shape)  
            dat.inputs=np.zeros((k*48+48,dec_.shape[0]))
            dat.targets=np.zeros((1,dec_.shape[0]))
            for i in range(dec_.shape[0]):
                dat.inputs[:,i]=dec_[i,-48:,:].ravel()
                dat.targets[0,i]=data[w+i]
            for Nl in Nls:
                for hidden in [10,20,40,80,160]:
                    for reg in [0.2,0.4,0.8]:
                        for iss in [0.01,0.1,1]:
                            hyper=[Nl,hidden,reg,iss,w,k]
                            p,_=edRVFL_predict(hyper, dat, train_idx, val_idx, s)
                            test_outputs=scaler.inverse_transform(p)
                            actuals=raw_data[-len(val_idx):]
                            # print(actuals.shape,test_outputs.shape)
                            test_err=compute_error(actuals,test_outputs,None)['RMSE']
                            if test_err<loss:
                                loss=test_err
                                best_hypers=hyper# print(best_hypers)
    return best_hypers

def ed_cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,scaler=None,s=0):
    # if len(hypers)!=Nl:
    #     print('Error')
    #     return None
    # cvloss=[]
    # for i in range(1,Nl):
    #     edhyper=hypers[:i+1]
    #     test_outputs_norm=edRVFL_predict(edhyper,data,train_idx,val_idx,s)
        
    #     test_outputs=scaler.inverse_transform(test_outputs_norm)
    #     actuals=raw_data[-len(val_idx):]
    #     test_err=compute_error(actuals,test_outputs,None)
    #     cvloss.append(test_err['RMSE'])
#    print(cvloss,cvloss.index(min(cvloss)))
    return hypers#[:cvloss.index(min(cvloss))+2]
    
def main():
    Nhs=np.arange(50,300,50)
    #np.arange(2,12,4)
    regs=[0]
    input_scale=[0.1]#,0.1,0.001]#[0.1,0.01,0.001]
    ratios=np.arange(0.05,1,0.05)
    deepRVFL_hypers=list(product(Nhs,regs,input_scale,ratios))
    order=48 
    seeds=10
    countrys=['SA','NSW','VIC','TAS']#['AT','BA','BE','BG','CH']
    year='2020'
    # hours=[str(i) for i in list(range(24))]
    #D:\AEMO\NSW\2020
    #PRICE_AND_DEMAND_202001_NSW1
    boat=50
    for co in countrys:
        for month_ in  ['01_','04_','07_','10_']:
            test_pres_ed=[]
            test_pres_ea=[]
            print(co+str(month_))
            name='PRICE_AND_DEMAND_2020'+month_+co+'1.csv'
            dataset='AEMO\\'+co+'\\'+year+'\\'+name
            df_data = pd.read_csv(dataset)
            data_=df_data['TOTALDEMAND'].values.reshape(-1,1)
            scaler=preprocessing.MinMaxScaler() 
            allpres=[]
            for s in np.arange(seeds):
                np.random.seed(s)
                val_l,test_l=int(0.1*data_.shape[0]),int(0.2*data_.shape[0])
                
                #cross validation 
                scaler.fit(data_[:-test_l-val_l])
                norm_data=scaler.transform(data_)
                # decomposer.ts_walkforward(x, w, k, 'ewt',pad_raw=True)
                # print(norm_data.shape)
                data=Struct()
                data.inputs,data.targets=format_data(norm_data,order)
                train_l=data.inputs.shape[1]-val_l-test_l
                train_idx=range(train_l)
                val_idx=range(train_l,train_l+val_l)
                test_idx=range(train_l+val_l,data.inputs.shape[1])
                # print(len(test_idx),test_l)
                best_hypers=cross_validation(deepRVFL_hypers[:],norm_data,data_[:-test_l],
                                                  train_idx,val_idx,Nls[0],regs,
                                                  input_scale,ratios=ratios,scaler=scaler,s=s,boat=boat)
                ed_best_hypers=ed_cross_validation(best_hypers,data,data_[:-test_l],
                                    train_idx,val_idx,Nls[0],scaler=scaler,s=s)
                print('Test')
                train_idx=range(train_l+val_l)
                scaler.fit(data_[:-test_l])
                norm_data=scaler.transform(data_)
                w,k=ed_best_hypers[-2],ed_best_hypers[-1]
            
                dec_=decomposer.ts_walkforward(norm_data.ravel(),w,k,'ewt',pad_l=0,pad_raw=True)
        
                data=Struct()
                
                data.inputs=np.zeros((k*48+48,dec_.shape[0]))
                data.targets=np.zeros((1,dec_.shape[0]))
                for i in range(dec_.shape[0]):
                    data.inputs[:,i]=dec_[i,:,-48:].ravel()
                    data.targets[0,i]=norm_data[w+i]
                # print(ed_best_hypers)
                test_outputs_norm_mea,alllayers=edRVFL_predict(ed_best_hypers,data,train_idx,test_idx,s)
                alllayers=scaler.inverse_transform(alllayers)
                allpres.append(alllayers)
                # test_outputs_ed=scaler.inverse_transform(test_outputs_norm_med)
                test_outputs_ea=scaler.inverse_transform(test_outputs_norm_mea)
                # test_pres_ed.append(test_outputs_ed)
                test_pres_ea.append(test_outputs_ea)
                actuals=data_[-test_l:]
                history=data_[:-test_l]
                # plt.figure()
                # plt.plot(actuals)
                # plt.plot(test_outputs_ea)
                # plt.show()
                test_err=compute_error(actuals,test_outputs_ea,history)
                print(test_err)
                print(len(ed_best_hypers))
            # test_p=np.concatenate(test_pres_ed,axis=1)
            # dfed=pd.DataFrame(test_p)
            # #D:\DeepRVFL-master\Results
            # dfed.to_csv('D:\\DeepRVFL-master\\Wind\\edRVFLmed'+loc+month_+'.csv')
            all_p=np.concatenate(allpres,axis=1)
            dfall=pd.DataFrame(all_p)
            dfall.to_csv('AEMO/ABA/allpEWTedRVFLGrid'+str(Nls[0])+str(boat)+name+'.csv')
            test_p=np.concatenate(test_pres_ea,axis=1)
            dfea=pd.DataFrame(test_p)
            # dfea.to_csv('D:\\DeepESN-master\\AEMO\\EWTRVFLBOA'+str(boat)+name)

                
                
if __name__ == "__main__":
    # dataset='D:\\UK load\\Monthly-hourly-load-values_2006-2015.csv'
    # df_data = pd.read_csv(dataset)
    # countrys=['AT','BA','BE','BG','CH']
    # year=2012
    # hours=[str(i) for i in list(range(24))]
    # for co in countrys:
    #     c_data=df_data.loc[(df_data['Country']==co)&(df_data['Year']==year)&(df_data['Month']==1)]
    #     data_=c_data[hours].values.reshape(-1,1)
    #     plt.figure()
    #     plt.plot(data_)
    #     plt.show()
    Nls=[12]
    decomposer=ForecastLib.TsPrep()
    main()
        
