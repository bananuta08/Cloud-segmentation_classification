## Detection and classification of ground-based cloud images
The algorithm contains two steps: threshold image segmentation to divide the images into areas of interest - clouds and background, and identification of cloud types, using Leave-one-out method, based on descriptors extracted from the segmented areas.
#### Segmentation
The image set used contains unimodal and bimodal images. Depending on the image type, the segmentation algorithm is divided into two sections: segmentation using a fixed threshold and segmentation using an adaptive threshold, unique for each image, using the MCE technique.
#### Classification
For the classification of cloud types into four distinct classes (Cumulus cloud class, Cirrus cloud class, Clear sky class and Stratus cloud class) the Euclidean distance between the feature vectors: FSC (fractional sky cover), CB (cloud brokenness) and TH (thickness) of the segmented test image and training images is determined. 
