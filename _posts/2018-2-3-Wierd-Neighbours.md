---
layout: post
title: Wierd Neighbours
description: A pecularity of k-Nearest-Neighbours
comments: true
---
####  A pecularity of k-Nearest-Neighbours
In this post, we will study the k-Nearest Neighbours (k-NN) classifier algorithm in unique circumstances to identify unexpected behaviors. In particular, we find a data generating function, where, contrary to the usual case, increasing k (for a range of k values) increases performance on the generated training set. On this set, we also explore different distance metrics and how they might contribute to the phenomenon. It is assumed the reader has some familiarity with k-NN.

## Part 1 - The Data
The data generating function used to train and test the kNN classifier is a skewed version of scikit-learn’s *make_moons* function. I’ve denoted the function as *make_galaxy* due to its alikeness to a spiral galaxy. An example of a dataset generated from this function along with the code is shown below.

![Image of Example Dataset](/images/posts/knn/exampledata.jpg)

As shown in the code below, the function skews the popular moons dataset through multiplying by the inverse of its covariance matrix.

```python
def make_galaxies(n_samples=100, shuffle=True, noise=None random_state=None):
    X, y = make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
    cov = np.cov(X, rowvar=False) 
    cov_inv = np.linalg.inv(cov)
    X = np.dot(X, cov_inv)/2 
    return X, y
```
Some key features of this dataset that will be important later when analyzing the performance of the kNN algorithm are the clustering of points at the curve in each spiral arm combined with the sparse distribution along the ends. The points along the ends are much closer to the cluster of the opposite color than they are to their own.

## Part 2 - The Metrics
The metrics used to evaluate the distance between each point were the L1 and L2-norms, along with higher p-norms including the Chebyshev distance (which is the limit as p → ∞). Visualizations of some p-norms are shown below.

![Visualizations of p-norms](/images/posts/knn/visualizations.jpg)

Notice the shapes of the different p-norms. These shapes are directly related to how the boundary for a point’s nearest neighbors (as measured by a specific p-norm) expands as k increases. We will see this later.

## Part 3 - Performance
In the usual case, one would expect that as *k* increases the performance of the classifier on the training set would decrease. Indeed, this is true when the L1-norm is used as the distance metric.

![Accuracy for L1-norm](/images/posts/knn/multiple_accuracy_p_1_.jpg)

Now, there still is an odd drop when *k* reaches about 90% of the dataset, but this still doesn’t violate the general idea that increasing k decreases training performance. The peculiarities occur when we examine the L2-norm.

![Accuracy for L2-norm](/images/posts/knn/multiple_accuracy_p_2_.jpg)

Wow, again, another oddity at around 90% except this one completely violates our general principle. Training performance goes **_up_** instead of down! How could this possibly be? Let’s visualize for a single point, how the k-NN algorithm chooses the point’s neighbors for various values of *k*. We want to examine how this decision boundary, determined by the L2-norm distance, expands as *k* increases.

![k-NN L2-norm](/images/posts/knn/knn_graph_pnt45_p2_mminkowski_t0.jpg)

It is slightly difficult to see but the boundary determined by the L2-norm expands radially outwards as visualized by **Figure 2**. We can see for k = 20, 40 and 60 (especially the last two), that this boundary engulfs the nearby cluster of the opposite color, causing the classifier to incorrectly predict the point as blue. It is only until around k = 90 that the cluster of the same color is within the boundary, causing there to be more points of the same color and thus a correct classification. I often like to visualize how the distance metric ”sees” the data by transforming the data. Since it is the L2-norm this would involve plotting the squared distance between the point of interest and the rest of the dataset across each feature so that the total distance measured by the L2-norm is simply the sum of the new coordinates.

![k-NN L2-norm transformed](/images/posts/knn/knn_graph_pnt45_p2_mminkowski_t1.jpg)

The boundary in this new coordinate system expands exactly like the L1-norm, since the distance becomes the sum of positive coordinates. Just imagine the expanding boundary as a diagonal line with negative slope being translated. In this visualization, we can see how the cluster of the same color is punished so harshly for being only slightly farther away than the local cluster of the opposite color. Unfortunately, normalizing the data fails to have a significant impact on this effect. This is because while the data is less skewed (and so the cluster and the end points are closer together) this underlying structure of clusters combined with sparsity is unaffected. 
> The key point here is that when the global structure (as described by distance between the points) of a dataset is more informative than the local structure as to what class a point belongs to, a larger *k* may be able to capture that more effectively than smaller *k*s.

![k-NN L2-norm normalized](/images/posts/knn/knn_graph_pnt45_p2_mminkowski_t0_norm.jpg)

![k-NN L2-norm normalized and transformed](/images/posts/knn/knn_graph_pnt45_p2_mminkowski_t1_norm.jpg)

## Part 4 - Other Metrics
The major difference between the L2-norm and higher p-norms is how the boundary expands as *k* increases. Depending on how the data is structured, different shaped decision boundaries may perform better. The performance differences for this particular dataset are outlined in the figure below. While slightly different, the same trend of increasing accuracy as *k* increases still occurs.

![Accuracy for multiple p-norms](/images/posts/knn/multiple_accuracy_p_1__2__5___Chebyshev__.jpg)

This visualization is for the L5-norm. It classifies the points in almost exactly the same way as the Chebyshev distance for this dataset since their boundaries are very similar.

![k-NN L5-norm](/images/posts/knn/knn_graph_pnt45_p5_mminkowski_t0.jpg)
