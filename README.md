# image_inpainti
This project contains the code for the image inpainting technique in the paper [1].
To run the project, please install the requiriment libraries by running following command
- pip install scipy matplotlib skimage

Currenlty, an implementation of the method is already available in the library skimage; however, there still exists two problem that I tried to solve in this problem
- The first problem is about the size of the linear system. There are often multiple tiny holes in an image and we need to set up an independent linear system to solve for unknown pixels in each hole. The size of these linear systems just need to be equal to the number of unknown pixels inside each hole; however, the current implementation generates linear systems whose sizes are equal to the number pixel of the whole image. 

- The second problem is regarding the formulation of the bi-laplacian matrix. Currently, bi-laplacian coefficients are calculated separately for each pixel while all of them, except ones next to the boundary, are the same.  
