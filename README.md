# Face Feature Reconstruction

This was made as part of an assignment for my Artificial Intelligence II course at Western University. It performs PCA (Principal Component Analysis) on a dataset of grayscale facial images stored in faces.dat. This process involves visualizing original and mean-centered images, compuiting eigenfaces (principal components), plotting eigenvalues, and reconstructing an image using different numbers of eigenfaces.

## Dataset

The dataset is expected to be named `faces.dat` where:
- Each row corresponds to a 64x64 grayscale facial image (flattened to 4096 values).
- Images are stored in row-major format.

## Example Output

### Face to be Reconstructed
![100th mean-centered](https://github.com/user-attachments/assets/caa9cc83-d0cf-41b1-b1e0-0801628481e5)

### Eigenfaces (Directions of variance)
![eigenfaces](https://github.com/user-attachments/assets/23afb615-b241-49ce-80b0-547556b75a05)

### Face Reconstruction
![recons](https://github.com/user-attachments/assets/a0983fc9-776b-4236-b7f6-836dbd5b3fa1)

### Comparison to Original
![compare](https://github.com/user-attachments/assets/17b9e613-0981-4a28-940b-87c4b2597db3)
