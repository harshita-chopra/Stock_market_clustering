# Stock market clustering
In this project, a k-means clustering model is trained to group companies with similar stock market movements over a period of three years.
The k-means clustering algorithm is part of the unsupervised learning family. It aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
1. The data is imported from Yahoo finance (754 rows) and daily movements are recorded.
2. Movements are normalized in a range of 0-1.
3. The normalized movements are fit into the k-means model and labels are predicted.
4. However, to visualize the clusters graphically, features (number of days here) need to be reduced. PCA is applied to achieve a linear dimensionality reduction using singular value decomposition of the data.
5. K-elbow visualizer with Silhouette score is used to obtain the optimal value of k on reduced data.
6. The two-dimensional PCA-reduced data is fit into the k-means model with 10 clusters and respective labels are predicted.
7. Two graphs showing clusters are plotted: one with scattered points and other with decision boundaries on colormap.

Outputs are shown below:


  ![K-elbow visualization](https://raw.githubusercontent.com/harshita219/Stock_market_clustering/master/output_images/kelbow.PNG)
  
  ![Labelling](https://raw.githubusercontent.com/harshita219/Stock_market_clustering/master/output_images/clusters.PNG)
   
  ![Clusters plotting 1](https://raw.githubusercontent.com/harshita219/Stock_market_clustering/master/output_images/plot1.PNG)
   
  ![Clusters plotting 2](https://raw.githubusercontent.com/harshita219/Stock_market_clustering/master/output_images/plot2.PNG)
