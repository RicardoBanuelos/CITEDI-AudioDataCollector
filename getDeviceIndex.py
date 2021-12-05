import pyaudio
audio = pyaudio.PyAudio()

# PyAudio va a encontrar muchos dispositivos, aqui vamos a buscar el nuestro
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i) 
    print(i," ",device_info['name']," ",device_info['defaultSampleRate'])
