# Face Feature Reconstruction

This was made as part of an assignment for my Artificial Intelligence II course at Western University. It performs PCA (Principal Component Analysis) on a dataset of grayscale facial images stored in faces.dat. This process involves visualizing original and mean-centered images, compuiting eigenfaces (principal components), plotting eigenvalues, and reconstructing an image using different numbers of eigenfaces.

## Dataset

The dataset is expected to be named `faces.dat` where:
- Each row corresponds to a 64x64 grayscale facial image (flattened to 4096 values).
- Images are stored in row-major format.
