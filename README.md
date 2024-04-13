## Detection and classification of ground-based cloud images
The algorithm contains two steps: threshold image segmentation to divide the images into areas of interest - clouds - and background, and identification of cloud types, using Leave-one-out method, based on descriptors extracted from the segmented areas.
The set of images from the used database includes 32 images of clouds that can be classified as Cumulus clouds, Cirrus clouds, clear sky or Stratus cloud images. There are two archives in this project that contain images: 2GT_img.rar and images.rar. images.rar contains 32 RGB images, while 2GT_img.rar contains 32 binary masks that correspond to the 32 original images. The binary masks are used to determine the accuracy of the segmentation algorithm that I choose to implement. 

#### Segmentation
The image set used contains unimodal and bimodal images. The present work follows the HYTA (Hybrid Thresholding Algorithm) algorithm, which combines fixed thresholding segmentation methods, which give good results for stratiform clouds and clear skies - unimodal images, with adaptive thresholding segmentation methods, which give good results for thin clouds - bimodal images. The following image represents the scheme of the hybrid cloud image segmentation algorithm.
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/49cb7946-acd0-4c3c-aadf-43cf4bf5ffb1)s
#### Classification
For the classification of cloud types into four distinct classes (Cumulus cloud class, Cirrus cloud class, Clear sky class and Stratus cloud class) the Euclidean distance between the feature vectors: FSC (fractional sky cover), CB (cloud brokenness) and TH (thickness) of the segmented test image and training images is determined. 
