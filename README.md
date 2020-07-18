# Breast Cancer Detection with Convolutional Neural Networks
Graduation Project group from Suez University - Faculty of Computers and Information
## introduction
Breast cancer is the most frequent cancer among women and the second most common cancer overall, impacting 2.1 million women each year, and also causes the greatest number of cancer-related deaths among women. In 2018, it is estimated that 627,000 women died from breast cancer – that is approximately 15% of all cancer deaths among women. While breast cancer rates are higher among women in more developed regions, rates are increasing in nearly every region globally. Breast Cancer is the most prevalent cancer among Egyptian women and constitutes 29% of National Cancer Institute cases. Median age at diagnosis is one decade younger than in countries of Europe and North America.  

Our goal is to build a convolutional neural networks that works on mammography images of breast cancer to make a binary classification  normal or abnormal.  
we are doing this to enhance the efforts done by our community leaders fighting the cancer and especially breast cancer like Baheya Hospital and Egyptian National Cancer Institute.

## Hands on Practice
we started this project with zero experience in data science, and to reach our goal we had to go through some steps.  
we had to learn about image processing, Ai, machine learning, neural networks and deep learning and what underlies them of technologies and tools like python, matlab, pytorch, keras and tensorflow.  
after being familiar with the concepts of machine learning and deep learning we started using these tools and technologies to create models and to train them on different datasets in order to increase our level of understanding of the concepts of neural networks.   
we divided ourselves to groubs, and every groub uses a different technology to measure and compare the performance between technologies and to determine which technology we are going to use in the whole project.  
#### matlab  
we tested an example with matlab using the version 2019b that comes with the deep learning toolbox that provides great visualization of the convolutional neural networks but we decided not to continue using it because somee complexities and errors we faced during testing.  
#### pytorch
another example we tested using pytorch, the network in the example will perform MNIST digit classification.  
the input images will be 28 x 28 pixel greyscale representations of digits. The first layer will consist of 32 channels of 5 x 5 convolutional filters + a ReLU activation, followed by 2 x 2 max pooling down-sampling with a stride of 2 (this gives a 14 x 14 output). In the next layer, we have the 14 x 14 output of layer 1 being scanned again with 64 channels of 5 x 5 convolutional filters and a final 2 x 2 max pooling (stride = 2) down-sampling to produce a 7 x 7 output of layer 2.  
After the convolutional part of the network, there will be a flatten operation which creates 7 x 7 x 64 = 3164 nodes, an intermediate layer of 1000 fully connected nodes and a softmax operation over the 10 output nodes to produce class probabilities. These layers represent the output classifier.  
 the network quite rapidly achieves a high degree of accuracy on the training set, and the test set accuracy and you can see the full implementation in **pytorch_convnet.py**.
 #### tensorflow
 after implementing a lot of examples using tensorflow we decided to create a model with the most famous example that all data scientists start with; the cats&dogs problem.  
 So, we created a model and we trained and tested it and the results was formidable to start with.  
 and you can see the architecture of the model and all the results of the training accuracy and the training loss as well as the validation accuracy and validation loss,   
 and you can see the full implementation in **catvsdogs.ipynb**.  
 so we decided to continue our project with tensorflow because its super performance and ease of implementation.  
 ## Dataset
 DDSM is a well-known dataset of normal and abnormal scans, and one of the few publicly available datasets of mammography imaging. Unfortunately, the size of the dataset is relatively small. To increase the amount of training data we extract the Regions of Interest (ROI) from each image, perform data augmentation and then train ConvNets on the augmented data. The ConvNets were trained to predict both whether a scan was normal or abnormal, and to predict whether abnormalities were calcifications or masses and benign or malignant.

The MIAS dataset is a very small set of mammography images, consisting of 330 scans of all classes. The scans are standardized to a size of 1024x1024 pixels. The size of the dataset made this unusable for training, but it was used for exploratory data analysis and as a supplementary test data set.

The University of California Irvine Machine Learning Repository contains several datasets related to breast cancer. These consist of one dataset which describes the characteristics of abnormalities and two which describe the characteristics of cell nuclei taken from fine needle biopsies. These were used for exploratory data analysis to gain insight into the characteristics of abnormalities.

The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The DDSM is saved as Lossless JPEGs, an archaic format which has not been maintained for several decades.

The CBIS-DDSM collection includes a subset of the DDSM data selected and curated by a trained mammographer. The CBIS-DDSM images have been pre-processed and saved as DiCom images, and thus are better quality than the DDSM images, but this dataset only contains scans with abnormalities. In order to create a dataset which can be used to predict the presence of abnormalities, the ROIs were extracted from the CBIS-DDSM dataset and combined with normal images taken from the DDSM dataset.

In order to create a training dataset of adequate size which included both normal and abnormal scans, images from the CBIS-DDSM dataset were combined with images from the DDSM dataset. While the CBIS-DDSM dataset included cropped and zoomed images of the Regions of Interest (ROIs), in order to have greater control over the data, we extracted the ROIs ourselves using the masks provided with the dataset.

For the CBIS-DDSM images the masks were used to isolate and extract the ROI from each image. For the DDSM images we simply created tiles of each scan and included them as long as they met certain criteria.
Both offline and online data augmentation was used to increase the size of the datasets.
### Training Datasets
Multiple datasets were created using different ROI extraction techniques and amounts of data augmentation. The datasets ranged in size from 27,000 training images to 62,000 training images.

Datasets 1 through 5 did not properly separate the training and test data and thus are not referenced in this work.
1.	Dataset 6 consisted of 62,764 images. This dataset was created to be as large as possible, and each ROI is extracted multiple times in multiple ways using both ROI extraction methods described below. Each ROI was extracted with fixed context, with padding, at its original size, and if the ROI was larger than our target image it was also extracted as overlapping tiles.

2.	Dataset 8 consisted of 40,559 images. This dataset used the extraction method 1 described below to provide greater context for each ROI. This dataset was created for the purpose of classifying the ROIs by their type and pathology.

3.	Dataset 9 consisted of 43,739 images. The previous datasets had used zoomed images of the ROIs, which was problematic as it required the ROI to be pre-identified and isolated. This dataset was created using extraction method 2 described below.

As Dataset 9 was the only dataset that did not resize the images based on the size of the ROI we felt that it introduced the least amount of artificial manipulation into the data and after it was created we focused on training with this dataset.

The CBIS-DDSM scans were of relatively large size, with a mean height of 5295 pixels and a mean width of 3131 pixels. Masks highlighting the ROIs were provided. The masks were used to define a square which completely enclosed the ROI. Some padding was added to the bounding box to provide context and then the ROIs were extracted at 598x598 and then resized down to 299x299 so they could be input into the ConvNet.
The ROIs had a mean size of 450 pixels and a standard deviation of 396. We designed our ConvNets to accept 299x299 images as input. To simplify the creation of the images, we extracted each ROI to a 598x598 tile, which was then sized down by half on each dimension to 299x299. 598x598 was just large enough that the majority of the ROIs could fit into it.

To increase the size of the training data, each ROI was extracted multiple times using the methodologies described below. The size and variety of the data was also increased by randomly horizontally flipping each tile, randomly vertically flipping each tile, randomly rotating each tile, and by randomly positioning each ROI within the tile.
## Training ConvNets
when we started evaluating the models we faced a big problem that is the processing power of our devices isn't suitable to run the training and it was a bootleneck for us   
while evaluating these models and the main reason for not trying a lot of models with different architectures that will need more and more processing power so we had to use the public cloud of google; google collaboratory known as google colab it provides us with GPU and RAM  that is able to train our models.  
we faced another problem when using google colab that is the GPU isn't always available and internet connection problems.  
We evaluated a large number of models on our dataset more than 30 models but the uploaded models here are the latest and the best results.  
we will demonstrate 2 models with thier architecture and graphs and results.  

### model_4_d.ipynb
#### Model description
We design the model based on known models the VGG model for example. The model is really deep With 9 convolution layers all with the same 3X3 filter with different numbers and different sequence Also has 5 max pooling layers and 3 dense layers. The model has 43,720,193 Trainable parameters.
Model training time is 2h:55m using google colab fast GPUs.
* Changing the optimizer to ADAM which is an adaptive learning rate optimization algorithm that’s been designed specifically for training deep neural networks. Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.  
* Weight balancing balances our data by altering the weight that each 	training example carries when computing the loss.

the pixel values in images are normalized between 0 and 1 in model_4_d. image normalization has a big impact when working with machine learning and deep learning algorithms. 


#### Model architecture

![Image of model_4](https://github.com/GP-FCI-SU/Breast-Cancer-Detection-with-Convolutional-Neural-Networks/blob/master/model_4.png)





 
 
 
 
