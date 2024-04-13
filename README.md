## Detection and classification of ground-based cloud images
The algorithm contains two steps: threshold image segmentation to divide the images into areas of interest - clouds - and background, and identification of cloud types, using Leave-one-out method, based on descriptors extracted from the segmented areas.
The set of images from the used database includes 32 images of clouds that can be classified as Cumulus clouds, Cirrus clouds, clear sky or Stratus cloud images. There are two archives in this project that contain images: 2GT_img.rar and images.rar. images.rar contains 32 RGB images, while 2GT_img.rar contains 32 binary masks that correspond to the 32 original images. The binary masks are used to determine the accuracy of the segmentation algorithm that I chose to implement.

![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/8f901d95-566a-4bc7-affd-236ce6eb7d0d)
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/2b095c60-d990-4c01-a8e2-9e759b462edb)
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/d7b103ac-1943-431a-b832-95574fc333ae)



#### Segmentation
The image set used contains unimodal and bimodal images. The present work follows the HYTA (Hybrid Thresholding Algorithm) algorithm, which combines fixed thresholding segmentation methods, which give good results for stratiform clouds and clear skies - unimodal images, with adaptive thresholding segmentation methods, which give good results for thin clouds - bimodal images. The following image represents the scheme of the hybrid cloud image segmentation algorithm.
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/49cb7946-acd0-4c3c-aadf-43cf4bf5ffb1)s
#### Classification
For the classification of cloud types into four distinct classes (Cumulus cloud class, Cirrus cloud class, Clear sky class and Stratus cloud class) the Euclidean distance between the feature vectors: FSC (fractional sky cover), CB (cloud brokenness) and TH (thickness) of the segmented test image and training images is determined. The following image represents the scheme of the classification algorithm:
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/9a73ce06-5eec-4bc5-9377-5b8bb4558c29)

Due to the limited database size, the leave-one-out cross-validation (LOOCV) method is suitable as a supervised classification algorithm for this work. This classification method involves splitting the training images and test images as follows: one testing image and the rest training images. The algorithm is run 32 times - the number of images in the database and each time the training image is different. To determine the class to which each test image belongs, the Euclidean distance between the feature vector of the training image (v_a) and the feature vector of the test image (v_t) will be calculated.
The minimum Euclidean distance determined between the training image and the test image will cause the test image to be assigned to the class of the training image. 

For easier manipulation and understanding of the classification results, a confusion matrix is used:
![image](https://github.com/ralucahabuc08/Cloud-segmentation_classification/assets/129282165/ddccd8cf-432b-4334-b7b4-bd9eee515313)
