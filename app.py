import argparse
import concurrent.futures
import csv
import datetime
import joblib
import numpy as np
import os
import pyaudio

from classify import classify_sound
from dBAlgorithm import get_rms, rms_to_decibels
from hmmlearn import hmm
from python_speech_features import mfcc
from scipy.io import wavfile

parser = argparse.ArgumentParser()
parser.add_argument("device_index",
                    help="Corre el script getDeviceIndex.py para encontrar este dato.",
                    type=int)
parser.add_argument("samp_rate",
                    help="Ingresa la taza de muestreo de tu dispositivo.",
                    type=int)
parser.add_argument("seconds", 
                    help="Ingresa cada cuantos segundos quieres que el dispositivo almacene informacion.",
                    type=int)
parser.add_argument("modelPath", 
                    help="Ingresa la ruta al modelo que deseas utilizar.")
args = parser.parse_args()

np.seterr(divide = 'ignore')

# Setup del archivo CSV

# Setup del stream de PyAudio
form_1 = pyaudio.paInt16                
chans = 1                               
samp_rate = args.samp_rate                      
duration = args.seconds   
chunk = samp_rate * duration
dev_index = args.device_index   

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__ == '__main__':

    hmm_models = joblib.load(args.modelPath)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format = form_1,
        channels = chans,
        rate = samp_rate,
        input_device_index = dev_index,
        input = True,
        frames_per_buffer = chunk
    )

    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while True:
                    print("Grabando...")
                    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames = np.frombuffer(data, dtype="int16")
                    rms = get_rms(data)
                    dB = rms_to_decibels(rms,95)
                    tag = None
                    if(dB > 20):
                        prediction = executor.submit(classify_sound, [frames, chans, audio, form_1, samp_rate, hmm_models])  
                        writer.writerow([date, dB, prediction.result()])
                    else:
                        writer.writerow([date, dB, tag])
        except KeyboardInterrupt:
            pass

    stream.close()
    audio.terminate()