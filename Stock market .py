import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas_datareader import data 
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans

companies_dict = {
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    'Navistar':'NAV',
    'Facebook':'FB',
    'Alibaba':'BABA',
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Mistubishi':'MSBHY',
    'Sony':'SNE',
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Ford':'F',
    'Bank of America':'BAC',
    'Dominos Pizza': 'DPZ',
    'Tesla':'TSLA',
    'Morgan Stanley': 'MS',
    'Proctor & Gamble': 'PG',
    'Oracle Corporation': 'ORCL',
    'Visa': 'V'}

data_source = "yahoo" 
start_date = datetime.date(2017,1,1) 
end_date = datetime.date(2019,12,31)
df = data.DataReader(list(companies_dict.values()), data_source,start_date,end_date)

# Numpy array of transpose of df['Open'] and df['Close'] prices to bring companies in rows
stockOpen = np.array(df["Open"]).T  
stockClose = np.array(df["Close"]).T 

# Recording change in prices or Movements
movements = stockClose - stockOpen
sum_movements = np.sum(movements, axis = 1)

print("\nFrom %s To %s " %(start_date, end_date))
print("\nCompany:  Sum of movements\n")

for i in range(len(companies_dict)):
    print(df['Open'].columns[i],sum_movements[i], sep = ":   ")

# Feature Scaling
scaler = preprocessing.Normalizer()
norm_movements = scaler.fit_transform(movements)

# Create Kmeans model using k=10
kmeans = KMeans(n_clusters = 10,max_iter = 1000,random_state=0)

# Fit normalized data to the kmeans model
kmeans.fit(norm_movements)
Klabels = kmeans.predict(norm_movements)
print("\n\nK-means inertia = ",kmeans.inertia_,"\n")

# Create dataframe to store companies and predicted labels
df1 = pd.DataFrame({"Companies":list(companies_dict.keys()), "Labels":Klabels }).sort_values(by=["Labels"],axis = 0)
display(df1)


# Reduce the dimensions using Principle Component Analysis for visualization
from sklearn.decomposition import PCA
reduced_data = PCA(n_components = 2).fit_transform(norm_movements)

# KElbowVisualizer method using Silhouette score
from yellowbrick.cluster import KElbowVisualizer
model = KMeans(random_state=0)
visualizer = KElbowVisualizer(model, k=(2,20), metric = 'silhouette',timings=False)
visualizer.fit(reduced_data)    
visualizer.poof()

# Create Kmeans model using k=10
kmeans = KMeans(n_clusters = 10,max_iter = 1000,random_state=0)
kmeans.fit(reduced_data)
Klabels = kmeans.predict(reduced_data)

# Create dataframe to store companies and predicted labels
print("\nK-means clustering using PCA reduced data...\n")
df2 = pd.DataFrame({"Companies":list(companies_dict.keys()), "Labels":Klabels }).sort_values(by=["Labels"],axis = 0)
display(df2)

# Plotting scattered points clusters

x_min, x_max = reduced_data[:, 0].min() - 0.1, reduced_data[:, 0].max() + 0.1
y_min, y_max = reduced_data[:, 1].min() - 0.1, reduced_data[:, 1].max() + 0.1

# Define Colormap
cmap = plt.cm.Paired

plt.clf()
plt.figure(figsize=(10,10))
for i in range(10):
    plt.scatter(reduced_data[Klabels==i,0],reduced_data[Klabels==i,1], c=[list(cmap.colors[i])], s=75, label='cluster '+str(i))

legd = plt.legend(frameon = True,fontsize=15)
legd.get_frame().set_edgecolor('k')
legd.get_frame().set_linewidth(2)

# Plot the centroid of each cluster as x
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker = "x",s = 169,linewidths = 3,color = "#808080",zorder = 10)
plt.title("K-Means scatter-points clustering on stock market movements (PCA-Reduced data)",fontsize=15)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()


# Plotting clusters on colormap

h = .001    # Step size of the mesh.
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 10))
plt.clf()
plt.imshow(Z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), 
           cmap=cmap,
           aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=15)
cb = plt.colorbar(orientation = 'horizontal')
cb.ax.tick_params(labelsize=15)
cb.set_label(label='Cluster labels',size=15)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clusters on Stock Market Movements (PCA-reduced data)\n'
          'Centroids are marked with white cross',fontsize=15)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
