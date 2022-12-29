# from scipy.io import wavfile
# from scipy.fftpack import fft
# from numpy import average, absolute
# import numpy as np

# LEFT = 0
# RIGHT = 1

# rate, data = wavfile.read('test.wav')

# SAMPLE_LENGTH = 0.5
# SAMPLE_COUNT = int(SAMPLE_LENGTH * rate)
# NUM_SAMPLES = int(len(data) / SAMPLE_COUNT)
# NUM_CHANNELS = 40

# maxItem = 0

# result = []
# for i in range(NUM_SAMPLES):
#   SAMPLE_START = i * SAMPLE_COUNT
#   SAMPLE_END = SAMPLE_START + SAMPLE_COUNT 
#   frequencies = absolute(fft(data[SAMPLE_START:SAMPLE_END,LEFT]))
#   channels = [0 for _ in range(NUM_CHANNELS)]
#   for i in range(NUM_CHANNELS):
#     sliceSize = int(len(frequencies) / NUM_CHANNELS)
#     sliced = frequencies[(sliceSize * i):(sliceSize * (i+ 1))]
#     channel = int(average(sliced))
#     channels[i] = channel
#     maxItem = max(channel, abs(maxItem))
#   result.append(channels)

# print ((np.array(result) / maxItem)[20])

import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

width = 1920
height = 1080
FPS = 24
seconds = 1
num_channels = 30
backgroundColor = 20
barColor = (10, 100, 250)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./noise.avi', fourcc, float(FPS), (width, height))

channels = np.random.random((num_channels))
channelWidth = width / len(channels)

for _ in range(FPS*seconds):
    channels = np.clip(((np.random.random((num_channels)) - 0.5) * 0.05) + channels, 0, 1)
    frame = np.full((height, width, 3), backgroundColor, dtype=np.uint8)
    for i, channel in enumerate(channels):
        cv2.rectangle(frame, (int(i * channelWidth), height), (int((i+1) * channelWidth), height - int(height * channel)), barColor, int(-1))
    video.write(frame)
video.release()