import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
def xVector(fileName):
    signal, fs = torchaudio.load(fileName)
    embeddings = classifier.encode_batch(signal)
    embeddings = embeddings.detach().cpu().numpy()
    embedding = embeddings[0][0]#size a 512
    return embedding

