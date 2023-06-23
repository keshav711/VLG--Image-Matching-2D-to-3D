# VLG--Image-Matching-2D-to-3D
Open project for VLG IITR.

Project description - The goal of this competition is to reconstruct accurate 3D maps using SfM techniqie.
***Participants are asked to estimate the pose for each image in a set with N images. Each camera pose is parameterized with a rotation matrix R and a translation vector T, from an arbitrary frame of reference.

Submissions are evaluated on the mean Average Accuracy (mAA) of the estimated poses. Given a set of cameras, parameterized by their rotation matrices and translation vectors, and the hidden ground truth, we compute the relative error in terms of rotation (ϵR, in degrees) and translation (ϵT, in meters) for every possible pair of images in N, that is, (N 2) pairs.

We then threshold each of this poses by its accuracy in terms of both rotation, and translation. We do this over ten pairs of thresholds: e.g. at 1 degree and 20 cm at the finest level, and 10 degrees and 5 m at the coarsest level. The actual thresholds vary for each dataset, but they look like this:

thresholds_r = np.linspace(1, 10, 10)  # In degrees.
thresholds_t = np.geomspace(0.2, 5, 10)  # In meters.
We then calculate the percentage of accurate samples (pairs of poses) at every thresholding level, and average the results over all thresholds. This rewards more accurate poses. Note that while you submit N, the metric will process all samples in (N 2) pairs.
Finally, we compute this metric separately for each scene and then average it to compute its mAA. These values are then averaged over datasets, which contain a variable number of scenes, to obtain the final mAA metric.

Submission - csv format
For each image ID in the test set, you must predict its pose. The file should contain a header and have the following format:

image_path,dataset,scene,rotation_matrix,translation_vector
da1/sc1/images/im1.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
da1/sc2/images/im2.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
etc
The rotation_matrix (a 3×3 matrix) and translation_vector (a 3-D vector) are written as ;-separated vectors. Matrices are flattened into vectors in row-major order. Note that this metric does not require the intrinsics (the calibration matrix K), usually estimated along with R and T during the 3D reconstruction process.***


Apporach - My approach was inspired  from (https://www.kaggle.com/code/bobfromjapan/imc-2023-submission-92nd-solution),I also used Trueprice's SQL handling.

I tried to optimize the solution including following techniques - 
Parameter tuning(min_pairs, num_features, resize_small_edge_to, sim_thresh).
Apply CLAHE(Contrast Limited Adaptive Histogram Equalization) to all images.
Use all reconstructions to get rotmat and tvec of images.
