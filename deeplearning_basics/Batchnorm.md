#### Overview
Training NN is complicated since the input distribution of each layer changes during training, due to parameter updates. This slows down the training since we require lower learning rates.
This is referred to as ***internal covariate shift***. BN tries to overcome this problem and has added advantages:
 * Training faster, lower epochs needed, since we can have higher learning rates
 * Acts as regularizer, in some cases which eliminates dropout requirement
 
 
#### Internal covariate shift
Change in the distribution of network activations due to change of internal parameters as we train the network. If we normalize the inputs to each layer, the training will converge faster.
Following from whitening of data i.e linearly transformed data by mean sub & divide by variance thie idea is to normalize the inputs to each layer.

Challenge is that if normalization is done while backprogpogating, the gradients also get normalized and network doesnt learn anything.
Specially for example, in case of non linear activation functions, this may lead to linear activations.
To solve this its necessary to take into account whole training and do some complex math (covariance matrix computation etc)

Solution:
 * ***normalize each feature separately*** instead of normalizing the layers input feature vector.
 * Learn parameters gamma and beta for each feature in batch norm layer using a ***Scale and Shift*** transformation (which is capable of learning identity function) i.e. gamma and beta are one.
 
 
#### High level algorithm (Forward pass):
 * For values of x over a minibatch
 * compute mean
 * compute variance
 * Do whitening
 * Scale and shift
 
So normalized activations can be seen as input to a subnetwork of linear transform.


High LR may cause gradients to vanish or explode, stuck in lower minima. BN helps since normalization does not allow small changes in parameters to explode the gradients.
