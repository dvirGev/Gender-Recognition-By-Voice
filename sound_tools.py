from pydub import AudioSegment
from pydub.playback import play
from os import listdir
import random
import json
import pandas as pd
import numpy as np
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier




def featurize(wavfile):
    #initialize features 
    hop_length = 512
    n_fft=2048
    #load file 
    y, sr = librosa.load(wavfile)
    #extract mfcc coefficients 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc) 
    #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
    mfcc_features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
                            np.mean(mfcc[1]),np.std(mfcc[1]),np.amin(mfcc[1]),np.amax(mfcc[1]),
                            np.mean(mfcc[2]),np.std(mfcc[2]),np.amin(mfcc[2]),np.amax(mfcc[2]),
                            np.mean(mfcc[3]),np.std(mfcc[3]),np.amin(mfcc[3]),np.amax(mfcc[3]),
                            np.mean(mfcc[4]),np.std(mfcc[4]),np.amin(mfcc[4]),np.amax(mfcc[4]),
                            np.mean(mfcc[5]),np.std(mfcc[5]),np.amin(mfcc[5]),np.amax(mfcc[5]),
                            np.mean(mfcc[6]),np.std(mfcc[6]),np.amin(mfcc[6]),np.amax(mfcc[6]),
                            np.mean(mfcc[7]),np.std(mfcc[7]),np.amin(mfcc[7]),np.amax(mfcc[7]),
                            np.mean(mfcc[8]),np.std(mfcc[8]),np.amin(mfcc[8]),np.amax(mfcc[8]),
                            np.mean(mfcc[9]),np.std(mfcc[9]),np.amin(mfcc[9]),np.amax(mfcc[9]),
                            np.mean(mfcc[10]),np.std(mfcc[10]),np.amin(mfcc[10]),np.amax(mfcc[10]),
                            np.mean(mfcc[11]),np.std(mfcc[11]),np.amin(mfcc[11]),np.amax(mfcc[11]),
                            np.mean(mfcc[12]),np.std(mfcc[12]),np.amin(mfcc[12]),np.amax(mfcc[12]),
                            np.mean(mfcc_delta[0]),np.std(mfcc_delta[0]),np.amin(mfcc_delta[0]),np.amax(mfcc_delta[0]),
                            np.mean(mfcc_delta[1]),np.std(mfcc_delta[1]),np.amin(mfcc_delta[1]),np.amax(mfcc_delta[1]),
                            np.mean(mfcc_delta[2]),np.std(mfcc_delta[2]),np.amin(mfcc_delta[2]),np.amax(mfcc_delta[2]),
                            np.mean(mfcc_delta[3]),np.std(mfcc_delta[3]),np.amin(mfcc_delta[3]),np.amax(mfcc_delta[3]),
                            np.mean(mfcc_delta[4]),np.std(mfcc_delta[4]),np.amin(mfcc_delta[4]),np.amax(mfcc_delta[4]),
                            np.mean(mfcc_delta[5]),np.std(mfcc_delta[5]),np.amin(mfcc_delta[5]),np.amax(mfcc_delta[5]),
                            np.mean(mfcc_delta[6]),np.std(mfcc_delta[6]),np.amin(mfcc_delta[6]),np.amax(mfcc_delta[6]),
                            np.mean(mfcc_delta[7]),np.std(mfcc_delta[7]),np.amin(mfcc_delta[7]),np.amax(mfcc_delta[7]),
                            np.mean(mfcc_delta[8]),np.std(mfcc_delta[8]),np.amin(mfcc_delta[8]),np.amax(mfcc_delta[8]),
                            np.mean(mfcc_delta[9]),np.std(mfcc_delta[9]),np.amin(mfcc_delta[9]),np.amax(mfcc_delta[9]),
                            np.mean(mfcc_delta[10]),np.std(mfcc_delta[10]),np.amin(mfcc_delta[10]),np.amax(mfcc_delta[10]),
                            np.mean(mfcc_delta[11]),np.std(mfcc_delta[11]),np.amin(mfcc_delta[11]),np.amax(mfcc_delta[11]),
                            np.mean(mfcc_delta[12]),np.std(mfcc_delta[12]),np.amin(mfcc_delta[12]),np.amax(mfcc_delta[12])])
    
    return mfcc_features


def add_512_features():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    MALES_PATH = r"\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\males"
    FEMALES_PATH = r"\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\females"
    MALES_OUT_PATH = r"\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\males_out"
    FEMALES_OUT_PATH = r"\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\females_out"
    male_files = listdir(MALES_PATH)
    female_files = listdir(FEMALES_PATH)
    random.shuffle(male_files)
    random.shuffle(female_files)
    min_amount = 1000
    boys = []
    girls = []
    count = 0
    for file in male_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = featurize(f"{MALES_PATH}/{file}")
            signal, fs =torchaudio.load(f"{MALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            boys.append(features.tolist() + embedding.tolist())
            sound = AudioSegment.from_wav(f"{MALES_PATH}/{file}")
            sound.export(f"{MALES_OUT_PATH}/{file}", format='wav')
            count += 1
    
    count = 0
    for file in female_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = featurize(f"{FEMALES_PATH}/{file}")
            signal, fs =torchaudio.load(f"{FEMALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            girls.append(features.tolist() + embedding.tolist())
            sound = AudioSegment.from_wav(f"{FEMALES_PATH}/{file}")
            sound.export(f"{FEMALES_OUT_PATH}/{file}", format='wav')
            count += 1

    print("boys: ", len(boys))
    print("girls: ", len(girls))

    json_obj = {"males": boys, "females": girls}
    with open('boys_girls_audio.json', 'w') as outfile:
        json.dump(json_obj, outfile)

if __name__ == '__main__':
    # convert_distort()
    # add_distortion()
    # cat_all_youtube_audio()
    # play(get_sec_from_combined_audio(55))
    # overlay_sec_on_all_audio()
    # take_gender_audios()
    add_512_features()
    # total_audio_time()
    # make_new_validated_file()  # time in seconds is 49883.83199999991
    # pass

# females:  16541.18400000001
# males:    25383.10400000012

# total hours: 11.645

# other_males: 78798.42000000016 -> 14673
# other_females: 35088.51600000023 -> 6472

# total hours: 31.635

# other_males_with_teens: 87298.81199999961
# other_females_with_teens: 47991.240000000485

# total hours: 37.98
