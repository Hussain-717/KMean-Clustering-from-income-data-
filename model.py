import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

df = pd.read_csv("income.csv")
df = df.dropna()
x = df.drop(columns=['Name'])
km = KMeans(n_clusters=3)
km.fit(x)
y = km.predict(x)
df['cluster'] = y
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.savefig('cluster_plot.png')
joblib.dump(km, "model.pkl")