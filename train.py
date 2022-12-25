import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, create_model
from PCA import indealcomp
import time



# load the dataset
X, y = load_data()

# split the data into training, validation and testing sets
data = split_data(X, y, test_size=0.15, valid_size=0.15)


# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=10, restore_best_weights=True)

batch_size = 32
epochs = 9572*4

"""Step 4"""
# performing preprocessing part
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
data["X_train"] = mm.fit_transform(data["X_train"])
data["X_valid"] = mm.transform(data["X_valid"])
data["X_test"] = mm.transform(data["X_test"])
"""Step 5"""
# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA

pca = PCA()

data["X_train"] = pca.fit_transform(data["X_train"])
data["X_valid"] = pca.transform(data["X_valid"])
data["X_test"] = pca.transform(data["X_test"])

vector_length = indealcomp(pca.explained_variance_ratio_, plot=False)
start_time = time.time()

pca = PCA(n_components=vector_length)

data["X_train"] = pca.fit_transform(data["X_train"])
data["X_valid"] = pca.transform(data["X_valid"])
data["X_test"] = pca.transform(data["X_test"])

# construct the model
model = create_model(vector_length= vector_length)
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

print("--- %s seconds ---" % (time.time() - start_time))
