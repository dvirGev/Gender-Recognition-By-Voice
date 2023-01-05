import os
import numpy as np
from test import extract_feature
from sklearn.metrics import confusion_matrix
import pickle as pk
def loadData(foledrName):
    X=[]
    y=[]
    dirent = os.listdir(foledrName)
    for dir in dirent:
        gender = 1 if dir == 'male' else 0
        files = os.listdir(foledrName + '/' + dir)
        for file in files:
            X.append(extract_feature(foledrName + '/' + dir + '/' + file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True).reshape(1, -1)[0])
            y.append(gender)
    X = np.array(X)
    y = np.array(y)
    totalData = len(X)
    maleSample = len([1 for i in y if i==1])
    print(f'total data is {totalData}')
    print(f'total male {maleSample}, total female {totalData - maleSample}')
    return X,y
    


if __name__ == "__main__":
    # load the saved model (after training)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    from utils import load_data, split_data, create_model
    import argparse
    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                    and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    file = args.file
    name = file
    with open('vector_length.txt', 'r') as f:
      vector_length = int(f.read())
    # construct the model
    model = create_model(vector_length=vector_length)
    # load the saved/trained weights
    model.load_weights("results/model.h5")
    
    X, y = loadData(file)
    
    sc = pk.load(open("sc.pkl", 'rb'))
    X = sc.transform(X)
    pca = pk.load(open("pca1.pkl", 'rb'))
    X = pca.transform(X)
    pca = pca = pk.load(open("pca2.pkl", 'rb'))
    X = pca.transform(X)
    # Get the true labels and predicted labels for the test set
    y_pred = [1 if i[0] > 0.5 else 0 for i in model.predict(X)]
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    # Print the confusion matrix
    print(conf_matrix)
    Accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / ((conf_matrix[0][0] + conf_matrix[1]
    [1] + conf_matrix[0][1] + conf_matrix[1][0])) 
    Recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    Precision = conf_matrix[1][1] / ((conf_matrix[0][1]) + conf_matrix[1][1])
    F1_score = 2*Recall*Precision / ((Recall + Precision))
    print(f'Accuracy {Accuracy*100:.2f}%')
    print(f'Recall {Recall*100:.2f}%')
    print(f'Precision {Precision*100:.2f}%')
    print(f'F1_score {F1_score*100:.2f}%')