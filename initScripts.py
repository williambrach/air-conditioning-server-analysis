import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as s
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

class LinRegressionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.linReg = LinearRegression()
        
    def fit(self, X=None, Y=None, y=None):
        x = X[Y]
        noNan = x[x[y].notnull()].drop(columns=y)
        noNanTinside = x[x[y].notnull()][y]
        self.linReg.fit(noNan,noNanTinside)
        return self

    def transform(self, X=None, Y=None, y=None):
        x = X[Y]
        nanT = x[x[y].isnull()].drop(columns=y)
        result = self.linReg.predict(nanT)
        X[y][X[y].isnull()] = result
        return X

def hourValueGraph(param=None,df=None):
    df_uci_hourly = df[[param]]
    df_uci_hourly['hour'] = df_uci_hourly.index.hour
    df_uci_hourly.index = df_uci_hourly.index.date
    df_uci_pivot = df_uci_hourly.pivot(columns='hour')
    df_uci_pivot = df_uci_pivot.dropna()
    df_uci_pivot.T.plot(figsize=(20,8), legend=False, color='blue', alpha=0.2)
    
def silhouetteScoreGraph(param=None,df=None):
    df_uci_hourly = df[[param]]
    df_uci_hourly['hour'] = df_uci_hourly.index.hour
    df_uci_hourly.index = df_uci_hourly.index.date
    df_uci_pivot = df_uci_hourly.pivot(columns='hour')
    df_uci_pivot = df_uci_pivot.dropna()
    sillhoute_scores = []
    n_cluster_list = np.arange(2,30).astype(int)
    X = df_uci_pivot.values.copy()
    # Very important to scale!
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    for n_cluster in n_cluster_list:    
        kmeans = KMeans(n_clusters=n_cluster)
        cluster_found = kmeans.fit_predict(X)
        sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    plt.plot(sillhoute_scores)
    
def kMeansGraphOfClusters(param=None,df=None,numOfClusters=3):
    df_uci_hourly = df[[param]]
    df_uci_hourly['hour'] = df_uci_hourly.index.hour
    df_uci_hourly.index = df_uci_hourly.index.date
    df_uci_pivot = df_uci_hourly.pivot(columns='hour')
    df_uci_pivot = df_uci_pivot.dropna()
    sillhoute_scores = []
    X = df_uci_pivot.values.copy()
    # Very important to scale!
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    kmeans = KMeans(n_clusters=numOfClusters)
    cluster_found = kmeans.fit_predict(X)
    cluster_found_sr = pd.Series(cluster_found, name='cluster')
    df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True )

    fig, ax= plt.subplots(1,1, figsize=(18,10))
    color_list = ['blue','red','green']
    cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())

    for cluster, color in zip(cluster_values, color_list):
        df_uci_pivot.xs(cluster, level=1).T.plot(
            ax=ax, legend=False, alpha=0.05, color=color, label= f'Cluster {cluster}'
            )
        df_uci_pivot.xs(cluster, level=1).median().plot(
            ax=ax, color=color, alpha=0.9, ls='--'
        )

def loadDataset(resampleParam):
    #load temperature and humidity data from outside
    outDf = pd.read_csv("./data/data.csv", delimiter = ";")
    outDf = outDf.rename(columns={'Tim': "Time", "Temperature": "T-outside", "Humidity": "H-outside"})
    outDf.index = pd.to_datetime(outDf['Time'], format='%d.%m.%Y %H:%M:%S' )
    outDf = outDf.drop(columns=['Time'])
    
    #load consumption data 
    consumptionDf = pd.read_csv("./data/spotreba.csv",delimiter = ";")
    consumptionDf = consumptionDf.rename(columns={'Tim': "Time", "delta kWh": "KWH"})
    consumptionDf.index = pd.to_datetime(consumptionDf['Time'], format='%d.%m.%Y %H:%M' )
    consumptionDf = consumptionDf.drop(columns=['Time'])
    
    #load temperature nad humidity data from inside
    inDf = pd.read_csv("./data/serverovna.csv",delimiter = ";")
    inDf = inDf.rename(columns={'Tim': "Time", "Temperature": "T-inside", "Humidity": "H-inside"})
    inDf.index = pd.to_datetime(inDf['Time'], format='%d.%m.%Y %H:%M:%S' )
    inDf = inDf.drop(columns=['Time'])
    
    #resample all data by input parameter
    outDfResampled = outDf.resample(resampleParam).mean()
    consumptionDfResampled = consumptionDf.resample(resampleParam).sum()
    inDfResampled = inDf.resample(resampleParam).mean()
    
    #join data
    df = outDfResampled.join(consumptionDfResampled,how="inner")
    df = df.dropna()
    df = df.join(inDfResampled,how="left")
    
    #fill nan from inside measurement by linear regression
    linReg = LinRegressionTransformer()
    linReg.fit(df,['T-outside','H-outside','T-inside'],'T-inside')
    df = linReg.transform(df,['T-outside','H-outside','T-inside'],'T-inside')
    linReg.fit(df,['T-outside','H-outside','H-inside'],'H-inside')
    df = linReg.transform(df,['T-outside','H-outside','H-inside'],'H-inside')
       
    return df