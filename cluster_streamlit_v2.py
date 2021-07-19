import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

df=pd.read_csv('df_temp.csv.zip',sep=';', compression='zip')
distanciadf=pd.read_csv('DISTANCIAjaccard159.csv.zip',sep=';', compression='zip')
distanciadf.drop('Unnamed: 0',axis=1,inplace=True)
distancia=distanciadf.values

processos=df.sort_values('CDPROCESSO').reset_index(drop=True)['CDPROCESSO'].unique()
st.sidebar.title('CLUSTERIZAÇÃO DE PROCESSOS')
st.sidebar.write('Jorge Luiz - EXECUÇÃO DE TÍTULO EXTRAJUDICIAL')

listaclasse=['NENHUM','DBSCAN','AVERAGE','COMPLETE','KMEANS']

paginaseleciona=st.sidebar.selectbox('Selecione o método de clusterização',listaclasse)



if paginaseleciona=='NENHUM':
    st.title('VISÃO GERAL DOS PROCESSOS POR SEMELHANÇA DE MOVIMENTAÇÕES')

    st.write(" JACCARD DISTANCE")
    st.write('Modern Multidimensional Scaling')

    fig,ax=plt.subplots(figsize=(10,10))
    sp =sns.scatterplot(x="X", y="Y", data=df, s=200,palette='tab10')
    st.pyplot(fig)

if paginaseleciona=='DBSCAN': 
    eps = st.sidebar.slider('Parâmetro de Clusterização', 0.0, 1.0, 0.45)
    st.title(paginaseleciona)
    clustering = AgglomerativeClustering(distance_threshold=eps,n_clusters=None,affinity='precomputed',linkage='single').fit(distancia)
    df['CLUSTER']=clustering.labels_
    df=df.astype({'CLUSTER': 'category'})
    fig,ax=plt.subplots(figsize=(10,10))
    sp =sns.scatterplot(x="X", y="Y",hue="CLUSTER",data=df, s=200,palette='tab10')
    plt.legend([],[], frameon=False)
    st.pyplot(fig)
    st.write("O número de processos são:",len(processos))
    st.write("O número de CLUSTERS são:",len(set(clustering.labels_)))
    st.write("NC/NP = ",len(set(clustering.labels_))/len(processos) )
    ncc=len(set(clustering.labels_))
    if (ncc!=1):
        l1=df.CLUSTER.value_counts(ascending=False).index[0]
        l2=df.CLUSTER.value_counts(ascending=False).index[1]

        
        st.write('MAIOR CLUSTER- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts())
        st.write('TAMANHO- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts().values.sum())
        st.write('SEGUNDO MAIOR CLUSTER- C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts())
        st.write('TAMANHO-C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts().values.sum())
        fig,ax=plt.subplots(figsize=(10,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('DISTRIBUIÇÃO DO GASTO DO TEMPO',fontsize=20)
        plt.xlabel('TEMPO (d)',fontsize=20)
        plt.xlim(0,4000)
        df[df['CLUSTER']==l1].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='MAIOR CLUSTER',c='blue')
        df[df['CLUSTER']==l2].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='SEGUNDO MAIOR',c='g')
        plt.ylabel('PROBABILIDADE',fontsize=20)
        st.pyplot(fig)
    
if paginaseleciona=='AVERAGE': 
    eps = st.sidebar.slider('Parâmetro de Clusterização', 0.0, 1.0, 0.45)
    st.title(paginaseleciona)
    clustering = AgglomerativeClustering(distance_threshold=eps,n_clusters=None,affinity='precomputed',linkage='average').fit(distancia)
    df['CLUSTER']=clustering.labels_
    df=df.astype({'CLUSTER': 'category'})
    fig,ax=plt.subplots(figsize=(10,10))
    sp =sns.scatterplot(x="X", y="Y",hue="CLUSTER",data=df, s=200,palette='tab10')
    plt.legend([],[], frameon=False)
    st.pyplot(fig)
    st.write("O número de processos são:",len(processos))
    st.write("O número de CLUSTERS são:",len(set(clustering.labels_)))
    st.write("NC/NP = ",len(set(clustering.labels_))/len(processos) )
    ncc=len(set(clustering.labels_))
    if (ncc!=1):   
        l1=df.CLUSTER.value_counts(ascending=False).index[0]
        l2=df.CLUSTER.value_counts(ascending=False).index[1]

        
        st.write('MAIOR CLUSTER- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts())
        st.write('TAMANHO- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts().values.sum())
        st.write('SEGUNDO MAIOR CLUSTER- C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts())
        st.write('TAMANHO-C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts().values.sum())
        fig,ax=plt.subplots(figsize=(10,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('DISTRIBUIÇÃO DO GASTO DO TEMPO',fontsize=20)
        plt.xlabel('TEMPO (d)',fontsize=20)
        plt.xlim(0,4000)
        df[df['CLUSTER']==l1].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='MAIOR CLUSTER',c='blue')
        df[df['CLUSTER']==l2].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='SEGUNDO MAIOR',c='g')
        plt.ylabel('PROBABILIDADE',fontsize=20)
        st.pyplot(fig)
    
if paginaseleciona=='COMPLETE': 
    eps = st.sidebar.slider('Parâmetro de Clusterização', 0.0, 1.0, 0.45)
    st.title(paginaseleciona)
    clustering = AgglomerativeClustering(distance_threshold=eps,n_clusters=None,affinity='precomputed',linkage='complete').fit(distancia)
    df['CLUSTER']=clustering.labels_
    df=df.astype({'CLUSTER': 'category'})
    fig,ax=plt.subplots(figsize=(10,10))
    sp =sns.scatterplot(x="X", y="Y",hue="CLUSTER",data=df, s=200,palette='tab10')
    plt.legend([],[], frameon=False)
    st.pyplot(fig)
    st.write("O número de processos são:",len(processos))
    st.write("O número de CLUSTERS são:",len(set(clustering.labels_)))
    st.write("NC/NP = ",len(set(clustering.labels_))/len(processos) )
    ncc=len(set(clustering.labels_))
    if (ncc!=1):
        l1=df.CLUSTER.value_counts(ascending=False).index[0]
        l2=df.CLUSTER.value_counts(ascending=False).index[1]

        
        st.write('MAIOR CLUSTER- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts())
        st.write('TAMANHO- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts().values.sum())
        st.write('SEGUNDO MAIOR CLUSTER- C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts())
        st.write('TAMANHO-C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts().values.sum())
        fig,ax=plt.subplots(figsize=(10,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('DISTRIBUIÇÃO DO GASTO DO TEMPO',fontsize=20)
        plt.xlabel('TEMPO (d)',fontsize=20)
        plt.xlim(0,4000)
        df[df['CLUSTER']==l1].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='MAIOR CLUSTER',c='blue')
        df[df['CLUSTER']==l2].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='SEGUNDO MAIOR',c='g')
        plt.ylabel('PROBABILIDADE',fontsize=20)
        st.pyplot(fig)
    
if paginaseleciona=='KMEANS':
    st.title(paginaseleciona)
    ncc=st.sidebar.text_input("Número de Clusters desejável")
    if ncc!='':
        nc=int(ncc)
        X=df[['X','Y']].values
        kmeans = KMeans(n_clusters=nc, random_state=42,n_init=100, max_iter=3000).fit(X)
        df['CLUSTER']=kmeans.labels_
        df=df.astype({'CLUSTER': 'category'})
        fig,ax=plt.subplots(figsize=(10,10))
        sp =sns.scatterplot(x="X", y="Y",hue="CLUSTER",data=df, s=200,palette='tab10')
        plt.legend([],[], frameon=False)
        st.pyplot(fig)
        st.write("O número de processos são:",len(processos))
        st.write("O número de CLUSTERS são:",len(set(kmeans.labels_)))
        st.write("NC/NP = ",len(set(kmeans.labels_))/len(processos) )
		
        l1=df.CLUSTER.value_counts(ascending=False).index[0]
        l2=df.CLUSTER.value_counts(ascending=False).index[1]

		
        st.write('MAIOR CLUSTER- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts())
        st.write('TAMANHO- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts().values.sum())
        st.write('SEGUNDO MAIOR CLUSTER- C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts())
        st.write('TAMANHO-C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts().values.sum())
        fig,ax=plt.subplots(figsize=(10,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('DISTRIBUIÇÃO DO GASTO DO TEMPO',fontsize=20)
        plt.xlabel('TEMPO (d)',fontsize=20)
        plt.xlim(0,4000)
        df[df['CLUSTER']==l1].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='MAIOR CLUSTER',c='blue')
        df[df['CLUSTER']==l2].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='SEGUNDO MAIOR',c='g')
        plt.ylabel('PROBABILIDADE',fontsize=20)
        st.pyplot(fig)
    else:
        nc=2
        X=df[['X','Y']].values
        kmeans = KMeans(n_clusters=nc, random_state=42,n_init=100, max_iter=3000).fit(X)
        df['CLUSTER']=kmeans.labels_
        df=df.astype({'CLUSTER': 'category'})
        fig,ax=plt.subplots(figsize=(10,10))
        sp =sns.scatterplot(x="X", y="Y",hue="CLUSTER",data=df, s=200,palette='tab10')
        plt.legend([],[], frameon=False)
        st.pyplot(fig)
        st.write("O número de processos são:",len(processos))
        st.write("O número de CLUSTERS são:",len(set(kmeans.labels_)))
        st.write("NC/NP = ",len(set(kmeans.labels_))/len(processos) )
		
        l1=df.CLUSTER.value_counts(ascending=False).index[0]
        l2=df.CLUSTER.value_counts(ascending=False).index[1]

        st.write('MAIOR CLUSTER- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts())
        st.write('TAMANHO- C1',df[df['CLUSTER']==l1].ASSUNTO.value_counts().values.sum())
        st.write('SEGUNDO MAIOR CLUSTER- C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts())
        st.write('TAMANHO-C2',df[df['CLUSTER']==l2].ASSUNTO.value_counts().values.sum())
        fig,ax=plt.subplots(figsize=(10,10))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('DISTRIBUIÇÃO DO GASTO DO TEMPO',fontsize=20)
        plt.xlabel('TEMPO (d)',fontsize=20)
        plt.xlim(0,4000)
        df[df['CLUSTER']==l1].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='MAIOR CLUSTER',c='blue')
        df[df['CLUSTER']==l2].TMAX.plot(kind='kde',alpha=0.5,lw=2,label='SEGUNDO MAIOR',c='g')
        plt.ylabel('PROBABILIDADE',fontsize=20)
        st.pyplot(fig)
    
    
    
    
    
    
    
    
