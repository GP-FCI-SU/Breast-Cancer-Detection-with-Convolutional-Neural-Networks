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
