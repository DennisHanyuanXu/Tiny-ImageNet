# Image Classification on Tiny ImageNet

## 1. Dataset
### 1. 1 Tiny ImageNet
Tiny ImageNet Challenge is the default course project for Stanford CS231N. It runs similar to the ImageNet challenge (ILSVRC). Tiny ImageNet has __200 classes__ and each class has __500 training images__, __50 validation images__, and __50 test images__. The images are down-sampled to __64 x 64__ pixels.  

Since the test images are not labeled, I use the validation set as test set to evaluate the models.

### 1. 2 Mnist
Mnist is also used here as a way of evaluating and testing models.

## 2. Models
It's always a good idea to start off with a simple architecture.

### 2. 1 Multi-Class SVM
#### 2. 1. 1 Structure
The SVM model contains a single linear layer that maps input images to label scores. For the sake of linear mapping, each __64 x 64 x 3__ image (RGB) in Tiny ImageNet is stretched to a single column and matrix multiplication is performed to get the score of each class.

#### 2. 1. 2 Loss Function
To measure the quality of the model, we use one of the most common loss functions for Multi-Class SVM: hinge loss. Its objective is for the correct class of each image to have a predicted score that is higher than the incorrect classes by a fixed 'margin'.  

For each sample image, given:  
  - <img src="/tex/eb2f7f520ae67f98226c974a23e046d2.svg?invert_in_darkmode&sanitize=true" align=middle width=46.93135754999999pt height=20.221802699999984pt/> (a vector of <img src="/tex/45e0c7580575cc1d2bfed5089d5f1c7d.svg?invert_in_darkmode&sanitize=true" align=middle width=39.504438599999986pt height=14.15524440000002pt/> values)
  - label <img src="/tex/99af0656694a7a485b0e272cbc1b1921.svg?invert_in_darkmode&sanitize=true" align=middle width=122.03460554999998pt height=24.65753399999998pt/>

We can define the loss function as:  
  - <img src="/tex/096e8125b7acc1878f82be135a249c53.svg?invert_in_darkmode&sanitize=true" align=middle width=365.2714361999999pt height=24.657735299999988pt/>

Where <img src="/tex/034134687001feaa0f24978b15341be5.svg?invert_in_darkmode&sanitize=true" align=middle width=119.04862529999997pt height=24.65753399999998pt/> and <img src="/tex/779fb067c540493cc039a6403e0ea58d.svg?invert_in_darkmode&sanitize=true" align=middle width=40.79630609999999pt height=22.831056599999986pt/>, <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/> is an integer usually set to <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> or <img src="/tex/76c5792347bb90ef71cfbace628572cf.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>. Basic hinge loss has <img src="/tex/012b36279aac832bdad672ff18d4243a.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/> while squared hinge loss, another commonly used loss function that has a stronger penalty for samples that violate margins, has <img src="/tex/4c0a4b4c466c9858130ec7facb8f2b8a.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/>.  

It's also normal to add regularization to the loss function so that the loss function can be based on both data and weights. But in this case, SVM has already struggled to fit the training data as a linear classifier, adding regularization only results in lower accuracy.

#### 2. 1. 3 Performance
The SVM model can reach __5%__ top-1 accuracy and __16%__ top-5 accuracy.

