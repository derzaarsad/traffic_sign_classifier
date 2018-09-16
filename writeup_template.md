# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_image/4_Speed-limit-70kmh.jpg "Traffic Sign 1"
[image5]: ./test_image/11_Right-of-way.jpg "Traffic Sign 2"
[image6]: ./test_image/12_Priority-road.jpg "Traffic Sign 3"
[image7]: ./test_image/14_Stop.jpg "Traffic Sign 4"
[image8]: ./test_image/15_No-vehicles.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/derzaarsad/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

This is the first time I've developed a CNN classifier from scratch and I've decided not to do a data augmentation so that I can get a feel of how to develop a neural network without altering the input data set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer because it combines stochastic gradient descent with moving average, keeping the gradient direction accurate even though a mini batch is used while training. I kept the epoch between
60 - 150 to avoid overfitting caused by too many training iterations. Actually, I could implement an early break by adding a loop break on the training loop after it reachs some value, but I want to keep the network to be trained
for a little while after it reachs some value to avoid stopping the training too early. So at the end I used a fix epochs number. For the batch size I used the biggest applicable size on my laptop, on my case it is 512.
The learning rate is constant 0.001 for the whole time.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I plotted an accuracy over time diagram to see the accuracy growth on the validation and training set after each training iteration. The difference of accuracy between the validation and training set gives a reference of
which approach should be taken to find the solution. The rule of thumb is: if there is a big gap between accuracy in validation and training set, then it could be that the classifier
suffers an overfitting. The causes for this overfitting can be whether that the architecture of the classifier is complex enough to memorize the correct output of the training input
or the classifier doesn't generalize very well because the training data don't vary much that. The other rule thumb is that if both of the accuracies on validation and training set
are bad, it means that the classifier architecture is too simple that it doesn't have enough parameter to classify the data.

I used LeNet 5 as a starting point on this project. I took the original architecture and trained the net with the following parameters:

Batch size = 128
Learning rate = 0.001
Epoch = 15

~~diagram~~

The end results were 0.874 accuracy on validation and 0.990 accuracy on training. These results indicate an overfitting, so I tried to add an L2 regularization with the penalty gain
(lambda error) of 0.001. I also increased the batch size to 512 to fully utilize the GPU memory that I had. The end results were as follows:

**Validation Accuracy = 0.838**
**Training Accuracy = 0.964**

At this point I was thinking that instead of regularize the training prematurely and causes underfitting, I should just try to achieve a very high training accuracy first and then
regularize it afterwards. For this reason I increased the complexity of the network by doubling each convolution filter. The end result of the training were as follows:

**Validation Accuracy = 0.860**
**Training Accuracy = 0.985**

the result was better but it still needed an improvement. I saw that many deep neural network uses residual block that allows the hidden layer to learn the "identity" function of the
input allowing more hidden layer to be stacked without losing the gradient by training (vanishing gradient). But ResNet is only practical if we have a very deep network and I don't
want to increase the depth of the network prematurely because it could increase the training and also the inferencing time of the classifier. Therefore I took the time to research on
the internet and found out that many people mixed a high level and low level feature by flattening all the neurons from each layer steps (normally after the pooling) and connected it
to a fully connected layer. So I applied this approach to my current net and got these end results:

**Validation Accuracy = 0.878**
**Training Accuracy = 0.994**

up until this point, the architecture is as follows:

~~architecture~~

Afterwards I increased the epoch to 60 to let the network be trained longer. Instead of preprocessing the input image into a specific color space, I added a 1x1 convolution filter to
allow the network to learn by itself, which colour channels are useful for this classification. The results were as follow:

**Validation Accuracy = 0.918**
**Training Accuracy = 0.999**

The accuracy was much better on the validation set. The accuracy of the training almost reached 1.000, it means that I could start to narrow the accuracy gap between validation and
training set. I increase the penalty gain of the L2 regularization with the expectation that the gap between the two accuracies is narrowed without compensating much on the training
accuracy. But unfortunately the results are complete opposite, the validation as well as training accuracy decreased slowly. One of the possible reason is that a bigger penalty by L2
regularization causes the weights of the CNN to become much smaller until a point where they are almost zero. Based on this reasoning, I removed the L2 regularization and used a dropout
regularization instead. The dropout probability that I used was 0.5 and the results were as follow:

**Validation Accuracy = 0.947**
**Training Accuracy = 0.991**

It gives us a much better accuracy and for this reason the dropout is kept for the regularization. The current architecture up until this point is as follows:

~~architecture~~

However, the validation accuracy of 0.947 doesn't guarantee that the test set accuracy is at least 0.93, it means that it is better to push the validation accuracy higher.

My plan was to add more convolution filters, making the whole network to have 3 levels of convolution layers that will be stacked together as a fully connected layer. To achieve
this, firstly I changed the filter from 5x5 to 3x3 to keep more outputs on the early layers. The size of the output of the first pooling is an odd number, to avoid padding I used 3
stride by the next pooling. And then I also add one more layer on the fully connected layer so that the amount of neurons on the hidden layers doesn't drop dramatically.

The architecture is as follows:

~~architecture~~

Results:

**Validation Accuracy = 0.943**
**Training Accuracy = 0.999**

The validation accuracy is comparable to the last training but the training is almost 1.000 so I kept the architecture. I tried max pool instead of average pool because people
normally uses max pool

**Validation Accuracy = 0.933**
**Training Accuracy = 0.999**

I increased the epoch again to 150.

**Validation Accuracy = 0.940**
**Training Accuracy = 1.000**

I saw that the validation accuracy stays for a long time in 0.94, so in my opinion it was still overfit. Therefore I took the epoch back to 60 so that the training breaks early and
then made the network simpler by removing one layer on the fully connected neurons that I had just added before. The results were as follows:

**Validation Accuracy = 0.950**
**Training Accuracy = 0.993**

I increased the epoch again to 150.

**Validation Accuracy = 0.957**
**Training Accuracy = 0.998**

the gap between validation and training accuracy was getting smaller but the network was a little bit under fit because the training accuracy was not 1.0, for this reason I added more
complexity to the network. Instead of adding a new fully connected layer, I added the depth of the existing convolution filters.

**Validation Accuracy = 0.978**
**Training Accuracy = 1.000**

It worked!! And the test accuracy is: **0.968**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4 =200x200] ![alt text][image5 =200x200] ![alt text][image6 =200x200] 
![alt text][image7 =200x200] ![alt text][image8 =200x200]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit 70kmh                     | Speed limit 70kmh   							| 
| Right-of-way at the next intersection | Right-of-way at the next intersection         |
| Priority road			                | Priority road									|
| Stop	      		                    | Stop					 				        |
| No vehicles			                | Yield      							        |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The classifier misclassify No vehicle sign to a Yield sign. If we look closely, both of the signs are similar with the difference
that one has a triangle form and the other is round. This kind of failure can be caused by an unbalanced data distribution in the training set between both signs. The effect is that the classifier did not learn enough about
how to differentiate between both signs and instead classify the input to a class that has more data in the training set because the classifier knows it better than the other. With this theory, it is not surprising that in our
case the classifier chose to classify the No vehicles input as Yield because the data distribution figure shows that the Yield sign data set is much bigger than the No vehicle sign data set

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99959     			| Speed limit 70kmh   							| 
| 0.99996424     		| Right-of-way at the next intersection 		|
| 1.					| Priority road									|
| 0.9999273	      		| Stop					 				        |
| 0.9999757				| No vehicles      							    |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


