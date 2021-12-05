# CITEDI-AudioDataCollector

Bibliotecas que se necesitan instalar en el RPi, se recomienda crear un virtual env e instalarlas ahi:
 pyaudio
 numpy
 wave
 hmmlearn
 scipy 
 python_speech_features
 
 Primero correr el script getDeviceIndex.py
 Este va a imprimir una lista de todos los dispositivos ya sean input/output de sonido, los va a imprimir de la sig manera:
 [index] [name] [samp_rate]
 
 Ya que el dispositivo de entrada sea localizado, anotar el indice y samp_rate
 
 Para ahora si correr el app.py
 python3 app.py indice samp_rate segundos modelo
 indice: Indice del dispositivo encontrado en el primer script
 samp_rate: Taza de muestreo del dispositivo encontrado en el primer script
 segundos: La cantidad de segundos que queremos grabar antes de escribir en el archivo csv.data 
 (Yo recomiendo max 5 segundos para que la clasificacion no reciba audios muy largos)
 modelo: Ruta al modelo HMM que se desea utilizar para clasificar
 
 En el archivo data.csv, el script va a escribir [año-mes-día-hora-minuto-segundo] [decibeles] [etiqueta]
 
 
 
