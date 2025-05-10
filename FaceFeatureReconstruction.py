import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('faces.dat')

image_200 = data[199, :]

#make 64x64 matrix
image_200 = image_200.reshape(64, 64)

#display image
plt.imshow(image_200, cmap='gray')
plt.title('200th Image')
plt.show()

#compute mean image
mean_image = np.mean(data, axis=0)

#subtract the mean from each image
data_centered = data - mean_image

image_100_centered = data_centered[99, :]

#display
image_100_centered = image_100_centered.reshape(64, 64)
plt.imshow(image_100_centered, cmap='gray')
plt.title('100th Mean-Centered Image')
plt.show()

#compute covariance matrix
cov_matrix = np.cov(data_centered, rowvar=False)

#eigenvalue decomposition
print('Eigenvalue Decomp in process, may take a minute...')
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#guarantee real eigenvalues
eigenvalues = np.real(eigenvalues)

#sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

#plot eigenvalues
plt.plot(eigenvalues, 'o-')
plt.title('Eigenvalues in Descending Order')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

#variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

#cumulative variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

#find number of components to keep
n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f'Dimensionality: {n_components}')

top_5_eigenvectors = eigenvectors[:, :5]

#guarantee real eigenvalues
top_5_eigenvectors = np.real(top_5_eigenvectors)

#display each eigenvector
for i in range(5):
    eigenface = top_5_eigenvectors[:, i].reshape(64, 64)
    plt.imshow(eigenface, cmap='gray')
    plt.title(f'Eigenface {i+1}')
    plt.show()

x = data_centered[99, :]

#reconstruct using top k components
K_values = [10, 100, 200, 399]
for K in K_values:
    #guarantee real eigenvalues
    top_K_eigenvectors = np.real(eigenvectors[:, :K])
    
    #project onto top k eigenvectors
    projection = np.real(np.dot(x, top_K_eigenvectors))
    
    #reconstruct image
    reconstructed = np.real(np.dot(projection, top_K_eigenvectors.T))
    
    #add back mean image
    reconstructed += mean_image
    
    # Display the reconstructed image
    reconstructed = reconstructed.reshape(64, 64)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'Reconstructed with {K} Components')
    plt.show()