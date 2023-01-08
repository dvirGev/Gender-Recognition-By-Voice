import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from fromSali import xVector
import pickle as pk

def convertFileToWav(file):
    # newName = file.replace('.ogg', '.wav')
    os.system(f"ffmpeg -i {file} test.wav")

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    # print(sample_rate)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        # print(mel)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    # print("the len of feat in the extract features " , len(result))
    result = np.append(result, xVector(file_name), axis=None)
    return result

if __name__ == "__main__":
    # load the saved model (after training)
    from utils import load_data, split_data, create_model
    import argparse
    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                    and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    file = args.file
    name = file
    # Open the file in read mode
    with open('vector_length.txt', 'r') as f:
      vector_length = int(f.read())
    # construct the model
    model = create_model(vector_length=vector_length)
    # load the saved/trained weights
    model.load_weights("results/model.h5")
    if not file or not os.path.isfile(file):
        raise ValueError("there is no file")
    convertFileToWav(file)
    # extract features and reshape it
    file = "test.wav"
    features = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True).reshape(1, -1)
    sc = pk.load(open("sc.pkl", 'rb'))
    features = sc.transform(features)
    pca = pk.load(open("pca1.pkl", 'rb'))
    features = pca.transform(features)
    pca = pca = pk.load(open("pca2.pkl", 'rb'))
    features = pca.transform(features)

    male_prob = model.predict(features)[0][0] 
    # model.predict(features)
    print(male_prob)
    female_prob = 1 - male_prob
    gender = "male" if male_prob >= female_prob else "female"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")

