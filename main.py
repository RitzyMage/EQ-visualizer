from scipy.io import wavfile
from scipy.fftpack import fft
from numpy import average, absolute
import numpy as np

LEFT = 0
RIGHT = 1

rate, data = wavfile.read('test.wav')

SAMPLE_LENGTH = 0.5
SAMPLE_COUNT = int(SAMPLE_LENGTH * rate)
NUM_SAMPLES = int(len(data) / SAMPLE_COUNT)
NUM_CHANNELS = 40

maxItem = 0

result = []
for i in range(NUM_SAMPLES):
  SAMPLE_START = i * SAMPLE_COUNT
  SAMPLE_END = SAMPLE_START + SAMPLE_COUNT 
  frequencies = absolute(fft(data[SAMPLE_START:SAMPLE_END,LEFT]))
  channels = [0 for _ in range(NUM_CHANNELS)]
  for i in range(NUM_CHANNELS):
    sliceSize = int(len(frequencies) / NUM_CHANNELS)
    sliced = frequencies[(sliceSize * i):(sliceSize * (i+ 1))]
    channel = int(average(sliced))
    channels[i] = channel
    maxItem = max(channel, abs(maxItem))
  result.append(channels)

print ((np.array(result) / maxItem)[20])

# print(rate, fft(data[1998:2000,LEFT]))