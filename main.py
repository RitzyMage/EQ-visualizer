from scipy.io import wavfile
from scipy.fftpack import fft
from numpy import average, absolute
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

LEFT = 0
RIGHT = 1


WIDTH = 1920
HEIGHT = 1080
SAMPLE_LENGTH = 0.5
NUM_CHANNELS = 40
FPS = 24
BACKGROUND_COLOR = 20
BAR_COLOR = (10, 100, 250)
CHANNEL_WIDTH = WIDTH / NUM_CHANNELS

rate, data = wavfile.read('test.wav')
SAMPLE_COUNT = int(SAMPLE_LENGTH * rate)
NUM_SAMPLES = int(len(data) / SAMPLE_COUNT)
SECONDS = len(data) / rate

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./noise.avi', fourcc, float(FPS), (WIDTH, HEIGHT))

print(FPS, SECONDS, FPS * SECONDS)
for ii in range(int(FPS*SECONDS)):
    channels = np.clip(((np.random.random((NUM_CHANNELS)) - 0.5) * 0.05), 0, 1)
    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
    for i, channel in enumerate(channels):
        cv2.rectangle(frame, (int(i * CHANNEL_WIDTH), HEIGHT), (int((i+1) * CHANNEL_WIDTH), HEIGHT - int(HEIGHT * channel)), BAR_COLOR, int(-1))
    video.write(frame)
video.release()

# for i in range(NUM_SAMPLES):
#   SAMPLE_START = i * SAMPLE_COUNT
#   SAMPLE_END = SAMPLE_START + SAMPLE_COUNT 
#   frequencies = absolute(fft(data[SAMPLE_START:SAMPLE_END,LEFT]))
#   channels = np.fill(NUM_CHANNELS, 0)
#   for i in range(NUM_CHANNELS):
#     sliceSize = int(len(frequencies) / NUM_CHANNELS)
#     sliced = frequencies[(sliceSize * i):(sliceSize * (i+ 1))]
#     channel = int(average(sliced))
#     channels[i] = channel
#   result.append(channels)


