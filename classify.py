import wave

from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

# Function that temporarily stores frames and classifies them 
def classify_sound(args):
    wf = wave.open("frames.wav", "wb")
    wf.setnchannels(args[1])
    wf.setsampwidth(args[2].get_sample_size(args[3]))
    wf.setframerate(args[4])
    wf.writeframes(b''.join(args[0]))
    wf.close()

    sampling_freq, sound = wavfile.read("frames.wav")

    mfcc_features = mfcc(sound, sampling_freq)
    max_score = -99999
    output_label = None

    for item in args[5]:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if score > max_score:
            max_score = score
            output_label = label
    if output_label == None:
        return "unknown"
    else: 
        return output_label