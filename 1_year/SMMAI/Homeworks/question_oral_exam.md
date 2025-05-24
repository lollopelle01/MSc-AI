Answer of [evangelista&#39;s questions](https://ai4climate.science/Notes/AI/SMMAI/Exam/SMMAI-questions-oral)

# Homework 1

## Linear Algebra

1) **Consider the random matrix. Describe the Behavior and relationship between $K_2(A)$ and $K_{\infty}(A)$. Why their overall trend is similar? Does the definition of ill-conditioning depend on the norm used? Is there a relationship between the condition number of a matrix and the relative error $E(x_{true},x)$ of the computed solution?**
   * Knowing that:

     * $K_2(A) := ||A||_2\cdot||A^{-1}||_2$
     * $K_\infty(A) := ||A||_\infty\cdot||A^{-1}||_\infty$

       Then we can notice that $\forall i |x_i| \le sup_j |x_j|$ but since $||x||_\infty := sup_j|x_j|$ we can obtain that
       $||x||_2 = \sqrt{\sum_i x_i^2} \le \sqrt{\sum_i \|x\|_\infty^2} = \sqrt{n||x||_\infty^2} = \sqrt{n}||x||_\infty$ and so the actual relation between the two norms will be $||x||_2 \le \sqrt{n}||x||_\infty$ .
       Then i also found the lower bound $\frac{||x||_\infty}{\sqrt{n}} \le ||x||_2$ ([ref](https://en.wikipedia.org/wiki/Matrix_norm)) .
       So, knowing that $\frac{||A||_\infty}{\sqrt{n}} \le ||A||_2 \le \sqrt{n}||A||_\infty$ and the same is true for $A^{-1}$ :

     - $\frac{||A||_\infty}{\sqrt{n}} \cdot \frac{||A^{-1}||_\infty}{\sqrt{n}} \le ||A||_2 \cdot ||A^{-1}||_2 \le \sqrt{n}||A||_\infty \cdot \sqrt{n}||A^{-1}||_\infty$
     - $\frac{K_\infty(A)}{n} \le K_2(A) \le n K_\infty(A)$

     Their overall trends are so similar because they have that strict relation.
   * Yes, since the definition only refers to a high or low conditioning number but it can have several different values with respect to the norm we use.
     For example, given a matrix like $A = \begin{bmatrix} 1 & 1000 \\ 0 & 1 \end{bmatrix}$ and its inverse $A^{-1} = \begin{bmatrix} 1 & -1000 \\ 0 & 1 \end{bmatrix}$
     We will get :

     * $K_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{1000}{1} = 1000 = 10^3$
     * $K_\infty(A) = \|A\|_\infty \cdot \|A^{-1}\|_\infty = 1000000=10^6$
       * $\|A\|_\infty = \max(\|[1, 1000]\|_\infty, \|[0, 1]\|_\infty) = 1000$
       * $\|A^{-1}\|_\infty = \max(\|[1, -1000]\|_\infty, \|[0, 1]\|_\infty) = 1000$
   * Yes, given a linear system $Ax=b$ and its small perturbation $  (A + \delta A)(x + \delta x) = b + \delta b $, then the relative error $E(x, \hat x)= \frac{\|x-\hat x \|}{\|x\|} = \frac{\|\delta x \|}{\|x\|}$ we can perform some steps:

     1. Since we know that $Ax=b$
        $\cancel{Ax} + A\delta x + \delta Ax +\delta A \delta x = \cancel{b} + \delta b$
     2. Since $\delta A \delta x \approx 0$
        $A \delta x = \delta b - \delta A x - \cancel{\delta A \delta x}$
     3. We isolate $\delta x$ and assume that $\exist A^{-1}$
        $\delta x = A^{-1}\delta b - A^{-1}\delta A x$
     4. We apply the norm and consider trinagle inequality and sub-multiplicative property of norms
        $\|\delta x\| = \|A^{-1}\delta b - A^{-1}\delta A x\| \le \|A^{-1}\| \|\delta b\| + \|A^{-1}\| \|\delta A\| \|x\|$
     5. Divide by $||x||$
        $ \frac{\|\delta x\|}{\|x\|} \le \|A^{-1}\| \frac{\|\delta b\|}{\|x\|} + \|A^{-1}\| \|\delta A\| = \|A^{-1}\| \cdot \Big(\frac{\|\delta b\|}{\|x\|} + \|\delta A\| \Big)$
     6. Consider $K(A)=||A|| \cdot ||A^{-1}||$
        $ \frac{\|\delta x\|}{\|x\|} \le \frac{K(A)}{\|A\|} \cdot \Big( \frac{\|\delta b\|}{\|x\|} + \|\delta A\|\Big) = K(A) \cdot \Big( \frac{\|\delta b\|}{\|A\|\|x\|} + \frac{\|\delta A\|}{\|A\|} \Big)$
     7. We can substitute $||A||\cdot||x||$ with $||b||$ since $||b|| = ||Ax|| \le ||A||\cdot||x||$
        $ \frac{\|\delta x\|}{\|x\|} \le K(A) \cdot \Big( \frac{\|\delta b\|}{\|b\|} + \frac{\|\delta A\|}{\|A\|} \Big)$

     As a consequence we can see that the accuracy depends on the condition number as an upper bound.

2. Consider the Vandermonde matrix. Describe the Behavior and relationship between $K_2(A)$ and $K_{\infty}(A)$. Why their overall trend is similar? Does the definition of ill-conditioning depend on the norm used? Is there a relationship between the condition number of a matrix and the relative error $E(x_{true},x)$ of the computed solution? **AS BEFORE**
3. Consider the Hilbert matrix. Describe the Behavior and relationship between $K_2(A)$ and $K_{\infty}(A)$. Why their overall trend is similar? Does the definition of ill-conditioning depend on the norm used? Is there a relationship between the condition number of a matrix and the relative error $E(x_{true},x)$ of the computed solution? **AS BEFORE**

# Homework 2

## SVD Decomposition (Dyads)

1. **Consider two different images. What do you observe if you compare the $k$ rank approximation of an image $X$ for increasing values of $k$? Is there a relationship between the meaningfulness of the dyad of $X$ for a given $k$ and the value of the associated singular value? What do you observe if you plot the approximation error $||X_k - X||_2$ compared with the plot of $\sigma_k$, for increasing values of $k$?**
   * By comparing the $k$-rank approximation of an image with the image itself you will notice that the approximation gets closer to the original as $k$ increases, since more small details (represented by less important dyads) are included in the image.
   * Larger singular values correspond to dyads that carry more significant information about the image structure, such as major features or contrasts.
   * You'll observe a rapid decreasing trend as $k$ increases, following a pattern where the first few singular values are much larger than the rest. This rapid decay means that a small number of singular values account for most of the image's structure.
2. **Consider an image $X$ and let $X_k$ be the $k$-rank approximation of $X$. What is the compression factor $c_k$? What is its behavior for increasing values of $k$? How does it relate with the visual quality of the image $X_k$? What is the approximation error when the compressed image requires the same amount of information as that of the uncompressed image (i.e., $c_k = 0$)?**
   * The compression factor $c_k = 1 - \frac{k(m+n+1)}{mn}$ is a measure of how much space we would save by using the $k$-rank approximation of an image $m \times n$ instead of the original image.
   * As $k$ increases $c_k$ decreases because more singular values and singular vectors are included in the approximation. This corresponds to less compression (since more information is stored to improve the approximation's quality).
   * When $c_k = 0$ we have an image that requires the same information of the original image since we don't save space anymore, so we can say we have an exact copy of the original image. Instead we notice that there is an approximation error of 1%, since the information of an image isn't the same as its graphical features.

## SVD Classification

1. **Consider the SVD Classification algorithm for the digits 3 and 4 on the MNIST dataset. Describe how it works. Discuss the obtained misclassification rate.**

   1. We extracted only the images/labels of the digit 3 and 4 from the datasets.
   2. We computed the SVD over those datasets.
   3. We used the obtained U1 and U2 (left singular vectors which represent the dominant features of each class) for reprojecting the spaces of each image. In this way we can represent the image in terms of the primary features of each class.
   4. At the end we computed both the distance among the projections and the image and the lowest indicates the classification.

   - The misclassification we got was very small: 0% for the train set and 0.5% for the test set. I think this happens because the features of the 2 classes are strongly different (for example 4 has a closed line while the 3 not, 4 is sharp while 3 is rounded) and so that the left singular vectors adequately represent the two classes. The result on the test set proves that the classfication is robust.
2. **Consider the SVD Classification algorithm for the digits 3 and 4 on the MNIST dataset. Compare the obtained misclassification rate on the training set and on the test set. Describe the concepts of underfitting and overfitting. Does the SVD Classification algorithm show overfitting/underfitting?**

   - The misclassification we got was very small: 0% for the train set and 0.5% for the test set.

   I think this happens because the features of the 2 classes are strongly different (for example 4 has a closed line while the 3 not, 4 is sharp while 3 is rounded) and so that the left singular vectors adequately represent the two classes. The result on the test set proves that the classfication is robust and so that the model generalizes well.

   **Underfitting** occurs when a model is too simplistic to capture the underlying patterns in the data. This leads to high error rates on both the training and test sets, as the model fails to learn the distinctions between classes effectively.

   **Overfitting** happens when a model learns the training data too well, capturing even noise and small variations specific to the training set. This often results in a low error rate on the training set but a significantly higher error on the test set, as the model struggles to generalize to new, unseen data.

   So we can see that the SVD classification considered does not neither underfit nor overfit.
3. **Repeat the experiment for different digits other than 3 or 4. Is there a relationship between the visual similarity of the digits and the classification error?**

   **???** We tried the same experiment with all the possible couples of digits and noticed that the overall misclassification is quite low with except for 0, which has a missclassification greater that 25% with all the digits (with except for 1). I explained myself this as a consequence of the fact that most of the digits have a curved or closed shape that are features of 0, while for example 1 is the only one that has none of them and didn't get misclassified. **???**
4. **Consider the SVD Classification algorithm for the digits 3 and 4 on the MNIST dataset. What happens to the accuracy when $k$ grows? Why does the accuracy over the test set not increase monotonically?**
   ... TODO ...
5. **Discuss the method to extend the SVD Classification algorithm to a 3-digit example. Discuss the results obtained by varying the combination of digits.**
   ... TODO ...

## Clustering with PCA

1. What is a Clustering algorithm? Explain how the PCA works to clusterize MNIST digits.

   * A clustering algorithm is a type of machine learning algorithm that groups data points into clusters, where points in the same cluster are more similar to each other than to points in different clusters.
   * The PCA is
     1. Compute the centroid of the dataset
     2. Center the dataset
     3. Compute the SVD of the centered dataset
     4. Truncate to k dimension the left singular vectors
     5. Project the  centered dataset into the reduced space
2. Compute the average distance of the centroid on the training and test set. Are there some differences?

... TODO ...

3. Define the classifier associated with PCA. Discuss the results.

... TODO ...

4. Define the classifier associated with PCA. What happens to the accuracy when $k$ grows? Does the accuracy over the test set not increase monotonically in $k$? Why?

... TODO ...

# Homework 3

## Gradient Descent (GD)

1. Comparison between GD with and without backtracking (for different $\alpha > 0$). What is the behavior for different functions? Explain.

...

2. By looking at the plots of $\|\nabla f(x^k)\|_2$, of the error $\|x_{\text{TRUE}} - x_k\|_2$ and of $\|x^k - x^*\|$, compare the convergence speed for different functions and for different values of $\alpha > 0$, constant and chosen with backtracking procedure.

...

3. Consider the function 1. By looking at the plots of $\|\nabla f(x^k)\|_2$, of the error $\|x_{\text{TRUE}} - x_k\|_2$ and of $\|x^k - x^*\|$, discuss the convergence by changing the starting iterate, the tolerances, and the step size.

...

4. Consider the function 2. By looking at the plots of $\|\nabla f(x^k)\|_2$, of the error $\|x_{\text{TRUE}} - x_k\|_2$ and of $\|x^k - x^*\|$, discuss the convergence by changing the starting iterate, the tolerances, and the step size.

...

5. Consider the function 3. By looking at the plots of $\|\nabla f(x^k)\|_2$, of the error $\|x_{\text{TRUE}} - x_k\|_2$ and of $\|x^k - x^*\|$, discuss the convergence by changing the value of $n$ as in the homework trace, the tolerances, and the step size.

...

6. Consider the function 4. By looking at the plots of $\|\nabla f(x^k)\|_2$, of the error $\|x_{\text{TRUE}} - x_k\|_2$ and of $\|x^k - x^*\|$, discuss the convergence by changing the value of $n$ as in the homework trace, the tolerances, and the step size.

...

7. Consider the function 5. Discuss the point of GD with different values of $x_0$ and different step-sizes. Observe when the convergence points the global minimum and when it stops on a local minimum or maximum.

...

## Stochastic Gradient Descent (SGD)

1. Discuss the behavior of the logistic regression classifier varying the training set dimension ($N_{\text{train}}$).

...

2. Discuss the behavior of the logistic regression classifier varying the two considered digits.

...

3. What are the differences at convergence of the parameters $w^*$ when computed by GD and SGD, in particular the error of $w^*$ against the true solution?

...

4. Compare the accuracy of the Logistic Regression Classifier against the SVD classifier defined above for the same considered digits (two digits only).

...

5. *Optional*: Compare the accuracy of the Logistic Regression Classifier against the SVD classifier defined above for three digits for the same considered digits.

...

# Homework 4 : MLE and MAP

1. What is the behavior of the trained regressor model $f_\theta(x)$, where $\theta$ is the MLE solution under Gaussian assumptions, for increasing values of $K$? Explain the plot where the training and the test error are compared for increasing values of $K$.

...

2. What is the behavior of the trained regressor model $f_\theta(x)$, where $\theta$ is the MAP solution under Gaussian assumptions, for increasing values of $K$ and fixed $\lambda$? Explain the plot where the training and the test error are compared for increasing values of $K$.

...

3. What is the behavior of the trained regressor model $f_\theta(x)$, where $\theta$ is the MAP solution under Gaussian assumptions, for fixed value of $K$ lower and/or greater than the true $K$ and different $\lambda$? Explain the plot where the training and the test error are compared for increasing values of $K$.

...

4. Comment the difference in relative error between the MLE and MAP solutions, for given $\lambda > 0$ and increasing $K$. What happens when $N$ increases, if everything else stays the same? What are the differences between the solution computed via GD, SGD and Normal Equations? Does the relative error between the computed weights and the true weights relates with the accuracy of the computed model? Explain.

...
