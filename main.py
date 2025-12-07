import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Read seed width vs length csv file, change data type to array
width_vs_length_df = pd.read_csv("seeds-width-vs-length.csv")
width_vs_length_array = pd.DataFrame(width_vs_length_df).to_numpy()

# Assign the 0th column of grains: width
width = width_vs_length_array[:,0]

# Assign the 1st column of grains: length
length = width_vs_length_array[:,1]

# Scatter plot xs vs ys
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(width_vs_length_array)

# Assign 0th column of pca_features: xs
width_PCA = pca_features[:,0]

# Assign 1st column of pca_features: ys
length_PCA = pca_features[:,1]

# Scatter plot of width and length after modifying using PCA
# Decorrelation is observed
plt.scatter(width_PCA, length_PCA)
plt.axis('equal')
plt.show()

# Make a scatter plot of the untransformed points
plt.scatter(width_vs_length_array[:,0], width_vs_length_array[:,1])

# Fit model to points
model.fit(width_vs_length_array)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Plot shows the first principal component, the direction in which the data varies the most.
plt.axis('equal')
plt.show()

# Generate array from fish data
fish_df = pd.read_csv("fish.csv")
fish_array = pd.DataFrame(fish_df).to_numpy()
fish_array = np.delete(fish_array, [0], axis=1)

# Create scaler
scaler = StandardScaler()

# Create a PCA instance
pca = PCA()

# Create pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(fish_array)

# Plot the explained variances. 2 features have significant variance. Apparent intrinsic dimension of 2.
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
