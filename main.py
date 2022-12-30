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
SAMPLE_LENGTH = 0.1
NUM_CHANNELS = 40
FPS = 24
BACKGROUND_COLOR = 20
BAR_COLOR = (10, 100, 250)
CHANNEL_WIDTH = WIDTH / NUM_CHANNELS
MAX_SAMPLE_VALUE = 1<<17

rate, data = wavfile.read('test.wav')
SECONDS = len(data) / rate
FRAME_COUNT = int(FPS*SECONDS)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./noise.avi', fourcc, float(FPS), (WIDTH, HEIGHT))

for frameIndex in range(FRAME_COUNT):
    # print("frame", frameIndex)
    time = frameIndex / FPS
    print('on frame', frameIndex, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
    sampleStartTime = time - (SAMPLE_LENGTH / 2)
    sampleEndTime = time + (SAMPLE_LENGTH / 2)
    sampleStartIndex = max(0, int(sampleStartTime * rate))
    sampleEndIndex = min(int(sampleEndTime * rate), len(data))
    # print("\tfrom", sampleStartTime, '=>', sampleEndTime, 'in seconds')
    # print("\tfrom", sampleStartIndex, '=>', sampleEndIndex, 'in audio indices')

    channels = np.full(NUM_CHANNELS, 0, dtype=float)
    frequencies = absolute(fft(data[sampleStartIndex:sampleEndIndex,LEFT]))
    for i in range(NUM_CHANNELS):
        sliceSize = int(len(frequencies) / NUM_CHANNELS)
        sliced = frequencies[(sliceSize * i):(sliceSize * (i+ 1))]
        channel = min(1, (float(average(sliced)) / MAX_SAMPLE_VALUE))
        channels[i] = channel
    # print('\tchannels', channels)

    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
    for i, channel in enumerate(channels):
        cv2.rectangle(frame, (int(i * CHANNEL_WIDTH), HEIGHT), (int((i+1) * CHANNEL_WIDTH), HEIGHT - int(HEIGHT * channel)), BAR_COLOR, int(-1))
    video.write(frame)
video.release()



