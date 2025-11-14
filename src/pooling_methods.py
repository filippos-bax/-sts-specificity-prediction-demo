import numpy as np

# Function to max pool embeddings from a list of lists
def max_pool_embeddings(list_of_lists):
    # Convert the list of lists to a NumPy array
    arr = np.vstack(list_of_lists)
    # Return the maximum value along the first axis (max pooling)
    return arr.max(axis=0).tolist()

# Function to mean pool embeddings from a list of lists
def mean_pool_embeddings(list_of_lists):
    # Convert the list of lists to a NumPy array
    arr = np.vstack(list_of_lists)
    # Return the mean value along the first axis (mean pooling)
    return arr.mean(axis=0).tolist()
