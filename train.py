import os
from preparation import extract_feature
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, create_model
from PCA import indealcomp, plotPCA
import pickle as pk
import time



# load the dataset
X, y = load_data()

# split the data into training, validation and testing sets
data = split_data(X, y, test_size=0.15, valid_size=0.15)


# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 32
epochs = 100 #66938 #9572*4


"""Step 4 - make the first sort of the vector, and   """
# # performing preprocessing part
# from sklearn.preprocessing import MinMaxScaler
# mm = MinMaxScaler()
# data["X_train"] = mm.fit_transform(data["X_train"])
# data["X_valid"] = mm.transform(data["X_valid"])
# data["X_test"] = mm.transform(data["X_test"])
# performing preprocessing part

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# data["X_train"] = sc.fit_transform(data["X_train"])
# data["X_valid"] = sc.transform(data["X_valid"])
# data["X_test"] = sc.transform(data["X_test"])


# """Step 5"""
# # Applying PCA function on training
# # and testing set of X component
# from sklearn.decomposition import PCA

# pca = PCA()

# data["X_train"] = pca.fit_transform(data["X_train"])
# data["X_valid"] = pca.transform(data["X_valid"])
# data["X_test"] = pca.transform(data["X_test"])

##plotPCA(pca.explained_variance_ratio_)

vector_length =705 # indealcomp(pca.explained_variance_ratio_) #705
# print(vector_length)
# start_time = time.time()

# pca = PCA(n_components=vector_length)

# data["X_train"] = pca.fit_transform(data["X_train"])
# data["X_valid"] = pca.transform(data["X_valid"])
# data["X_test"] = pca.transform(data["X_test"])

import pickle as pk
"""This save our how to save the pca model and calc"""
# pk.dump(sc, open("sc.pkl","wb"))
# pk.dump(pca, open("pca.pkl","wb"))


# pk.dump(pca, open("pca.pkl","wb"))
# construct the model
model = create_model(vector_length = vector_length)
# train the model using the training set and validating using validation set
model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])

# save the model to a file
model.save("results/model.h5")

# evaluating the model using the testing set
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

# print("--- %s seconds ---" % (time.time() - start_time))

n_male_samples = len([i for i in y if i == 1])
# get total female samples
n_female_samples = len([i for i in y if i == 0])
test_male_samples = len([i for i in data["y_test"] if i == 1])
test_female_samples = len([i for i in data["y_test"] if i == 0])
train_male_samples = len([i for i in data["y_train"] if i == 1])
train_female_samples = len([i for i in data["y_train"] if i == 0])
print(f'total male: {n_male_samples} female: {n_female_samples}')
print(f'test male: {test_male_samples} female: {test_female_samples}')
print(f'train male: {train_male_samples} female: {train_female_samples}')

from sklearn.metrics import confusion_matrix

# Get the true labels and predicted labels for the test set
y_true = data["y_test"]

y_pred = [1 if i[0] > 0.5 else 0 for i in model.predict(data["X_test"])]

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print(conf_matrix)