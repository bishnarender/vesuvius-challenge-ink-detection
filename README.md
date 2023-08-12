## vesuvius-challenge-ink-detection
## score at 5th position is achieved.


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
