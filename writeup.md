# **Traffic Sign Recognition** 

## Writeup
---
[//]: # (Image References)

[validation_set_counts]: ./writeup_images/validation_set_counts.png 
[training_set_counts]: ./writeup_images/training_set_counts.png
[test_set_counts]: ./writeup_images/test_set_counts.png
[prenormalized]: ./writeup_images/prenormalized.png
[postnormalized]: ./writeup_images/postnormalized.png
[image1]: ./5_test_images/20kmph-sign.jpg
[image2]: ./5_test_images/do-not-enter.jpg
[image3]: ./5_test_images/do-not-pass.jpg
[image4]: ./5_test_images/german-yield-sign.jpg
[image5]: ./5_test_images/traffic-light-sign.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nkowdley/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

After unpickling and loading the data, I used the lengths of the arrays to find the number of training, validation, and testing examples.  I used the pandas shape function to find the shape of the image data, and the numpy unique function to determine the number of classes.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

To get an idea of what the data set looked like, I first printed out 5 random images, along with the corresponding number(or label) to the sign.  This gave me some indication of what each image looked like and what needed to be done to it. 

Then, I plotted out the frequency of each label or sign in the training data, the validation data, and the test data.  This data is below:


![validation_set_counts]

![training_set_counts]

![test_set_counts]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I only normalized the data here by using (pixel-128.)/128.  This made my data easier to train by making it so that it had mean zero and equal variance.   Here are the pictures of how the data changed.  Note that the first picture is before and the second picture is after the normalization. 

![prenormalized]

![postnormalized]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model started out with the LeNet Lab, with a few tweaks to make it work with this data.  I also added dropout with a .9 keep probability, after the fully connected layers.

The layers in order were as follows:

| Layer         		|
|:---------------------:|
| Input         		|
| Convolution        	|
| RELU					|
| Max pooling	      	|
| Convolution   	    |
| RELU          		|
| Max Pooling			|
| Flatten               |
|Fully Connected        |
| RELU          		|
| Dropout         		|
|Fully Connected        |
| RELU          		|
| Dropout         		|
|Fully Connected        |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

N_CLASSES = 43
MU = 0
SIGMA = .1
EPOCHS = 100
BATCH_SIZE = 128
KEEP_PROB = .9
RATE  =  0.001

I chose a high number of epochs in order to garauntee my models accuracy. I also picked a high keep probability since trial and error showed me that a higher probability resulted in a better model.

My optimizer was the AdamOptimizer, which I pulled from the Lenet Lab.  


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.949
* test set accuracy of 0.935

If a well known architecture was chosen:
* What architecture was chosen?
    For this Lab I used the LeNet Architecture from the LeNet Lab.  I chose this as my starting point since I was very familiar with that architecture after the lab. There were a few modifications that I had to make to use this new data set. I also added dropout as a tuning measure after the fully connected layers, but I found that dropout did not significantly help my model work, hence why I used a high keep probability.
* Why did you believe it would be relevant to the traffic sign application?
    I thought this architecture would work well since it shared a few key similarities to the MNIST data in that there were several examples of each type of sign and signs, much like numbers have distinctive shapes and characteristics.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The Final Model had a high validation and test set accuracy (>93%). This proved to me that my model was working well and could accurately predict traffic signs.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

I found these images on the web, and preprocessed them in order to be immediately usable by first cropping/resizing the images on my computer locally, and then normalizing them using the same code I did on the pickled data.  These images might be hard to classify since resizing the images caused some of them to warp their proportions, and the resolution on these images suffered greatly from this.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20kph sign      		| Ahead only   									| 
| No Entry     			| No entry 										|
| Yield					| Yield											|
| No passing	      	| No passing					 				|
| Traffic Signals		| No passing for vehicles over 3.5 metric tons     |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.  This is not a favorable result, since 2 of the signs were wildly incorrect.  Looking at the images, There is heavy pixelation in the 20kph sign, and the traffic signals sign, which might explain why the model had trouble with them.  In particular, the signals in the Traffic Signals Sign are very hard to see, so it seems logical that the model would struggle with them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Here are the results of running my test 5 images through my Neural Network.  Note that for each string, the probability is listed below it.

Filename: traffic-light-sign.jpg
Top 5: [10  9 16 34 12]
Top 5 Strings and Probs:
No passing for vehicles over 3.5 metric tons
1.0
No passing
1.3791089e-08
Vehicles over 3.5 metric tons prohibited
3.7068947e-14
Turn left ahead
1.848368e-16
Priority road
5.4262295e-18

Filename: german-yield-sign.jpg
Top 5: [13 12 25 42  1]
Top 5 Strings and Probs:
Yield
1.0
Priority road
7.6121545e-09
Road work
6.862081e-10
End of no passing by vehicles over 3.5 metric tons
1.7625693e-11
Speed limit (30km/h)
3.9517136e-15

Filename: 20kmph-sign.jpg
Top 5: [35 36 38 34 33]
Top 5 Strings and Probs:
Ahead only
0.9983011
Go straight or right
0.0010695207
Keep right
0.0006130815
Turn left ahead
1.36279305e-05
Turn right ahead
2.489008e-06

Filename: do-not-enter-sign.jpg
Top 5: [17 14 11 29  0]
Top 5 Strings and Probs:
No entry
1.0
Stop
3.8185468e-31
Right-of-way at the next intersection
5.761884e-34
Bicycles crossing
2.3121945e-35
Speed limit (20km/h)
0.0

Filename: do-not-pass-sign.jpg
Top 5: [ 9 41 23 37 20]
Top 5 Strings and Probs:
No passing
1.0
End of no passing
3.2488647e-17
Slippery road
9.554499e-21
Go straight or left
2.9791922e-22
Dangerous curve to the right
6.605565e-23

Looking at these results I am amazed that 4/5 of my images actually turned up with 100%, seeing as 1 of these is clearly wrong(I had 3/5 correct).  The incorrect one is the traffic lights, which is very curious to me since it looks nothing like the No passing for vehicles over 3.5 metric tons sign.  The 20kmph sign was also interesting to me, as the probability there was 99.8% which was also high, and also looks nothing like the Ahead Only sign.  In fact, for both of these wrong images, the correct label was not even in the top 5! This to me suggests that either my 5 found images are nothing like the training/test data .