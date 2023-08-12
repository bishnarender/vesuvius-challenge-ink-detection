## vesuvius-challenge-ink-detection
## got a score higher than the winner.
![vesuvius-submission](https://github.com/bishnarender/vesuvius-challenge-ink-detection/assets/49610834/d3b84aaf-a22d-43f6-9637-c96fa0682ddf)

### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. all_in_one.ipynb
3. train_type_1.ipynb
4. vesuvius-submission.ipynb

<b>Code has been explained in the above files and in the linked files to these.</b>

The second numbered fragment (from three) is further divided into 3 fragments i.e. 2, 4 and 5. Thus, making a total of 5 fragments i.e. 1, 2, 3, 4 and 5. The division is based on image height i.e., by dividing the height in three equal portions. The second numbered fragment is chosen because of the large size of its contents.

Further, the 5 fold data is designed from 5 fragments. Let's see how. The volume contents (from each fragment) have been concatenated into shape [65, H, W] i.e. 65 is the number of tif images present in volume. This shape is further divided into patches of H=32 and W=32 i.e, of shape [65,32,32]. Along these the respective "mask, ir and label" images are also divided into patches of shape [32,32].

### train_patch_based
-----
Model input is fabricated by fixing 8 patches of shape [65,32,32] to a numpy array of shape [65, 256, 256]. Instead of feeding all layers to model only the middle 27 layers (19 to 47) is feeded i.e., in the shape [27,256,256], means feeding only 27 slices from z-direction. Similarly, the corresponding label is fabricated from patches to shape [256,256].

As the model output is in shape [8,8] so to calculate loss, the original resolution of the label/target has been scaled down (to 1/32 via interpolate) to convert  shape [256,256] to [8,8]. 

Following "timm" models have been used with different configuration of "middle layers, number of 3D CNN blocks and number of channels":
1. convnext_tiny.fb_in1k
2. resnetrs50
3. swinv2_tiny_window8_256.ms_in1k

ResidualConv3d Block is referred to as 3D CNN block and its count is changed by the arg parameter "--num_3d_layer".

"timm" model is referred to as 2.5D CNN model. This timm model is called 2.5D CNN because each 2D slice in a sample has the information of several adjacent slices. But the model is a normal 2D CNN.

![model_1](https://github.com/bishnarender/vesuvius-challenge-ink-detection/assets/49610834/49c2f2df-badb-4018-ae93-9c90f7a17068)

### train_patch_based_1
-----
Model input is fabricated by fixing 6 patches of shape [65,32,32] to a numpy array of shape [65, 192, 192]. Instead of feeding all layers to model only the middle 24 layers (20 to 45) is feeded i.e., in the shape [24,192,192], means feeding only 24 slices from z-direction. Similarly, the corresponding label is fabricated from patches to shape [192,192].

As the model output is in shape [6,6] so to calculate loss, the original resolution of the label/target has been scaled down (to 1/32 via interpolate) to convert  shape [192,192] to [6,6]. 

Following "backbone" models (of ResNet3dCSN) have been used with different configuration of "middle layers":
1. resnet50-irCSN
2. resnet152-irCSN

ResNet3dCSN Block is referred to as 3D CNN block.

![model_2](https://github.com/bishnarender/vesuvius-challenge-ink-detection/assets/49610834/32d4b673-16dc-443e-8695-6cf9c42f232a)

####  special augmentation (in addition to normal augmentation)
-----
1. Manifold mixup (mixup data after the stage of feature extraction).
2. cutmix augmentation (mixup cutted portion with flipped data).

#### What is nelder-mead algorithm (been used here for getting the best threshold) ?
-----
The <b>Nelder–Mead method (also downhill simplex method, amoeba method, or polytope method)</b> is a numerical method used to find the minimum or maximum of an objective function in a multidimensional space. It is particularly useful when the function is non-linear, and its derivatives might be difficult to compute or unavailable.
A starting point must be provided to the method/algorithm, which may be the endpoint of another global optimization algorithm or a random point drawn from the domain.
Choose an initial set of points (vertices) in the parameter space. These points form the simplex. Simplex is a shape structure composed of n + 1 points (vertices), where n is the number of input dimensions to the function.
Evaluate the objective function at each vertex of the simplex and order the vertices based on their function values. 
1. <b>Reflect:</b> calculate the centroid of all vertices except the worst (highest value) vertex and obtain a new point. The new point is obtained by reflecting the simplex along the line joining the worst vertex with the centroid. If the reflected point is better than the second-worst point but not better than the best, replace the worst (highest value) vertex with the reflection. 

2. <b>Expand:</b> If the reflected point is better than the best vertex, try expanding further along that direction. If the expansion point is even better, replace the ongoing point with the expansion.

3. <b>Contract:</b> If the reflected point is worse than the second-worst vertex, contract the simplex towards the better vertices. There are two types of contractions: outside and inside. An outside contraction replaces the worst point with a point between the <u>reflected point</u> and the <u>centroid</u>. An inside contraction replaces the worst point with a point between the <u>worst point</u> and the <u>centroid</u>.

4. <b>Shrink:</b> If none of the above steps result in a better vertex, perform a shrink operation by moving all other vertices toward the best vertex.
    
Continue iterating these steps until a termination criterion is met, such as a predefined number of iterations or when the simplex becomes sufficiently small.

![nelder_mead](https://github.com/bishnarender/vesuvius-challenge-ink-detection/assets/49610834/18713adf-02cd-42dc-bb46-7328884a849c)<br>
[Image Reference](https://capsis.cirad.fr/capsis/documentation/optimisation)



