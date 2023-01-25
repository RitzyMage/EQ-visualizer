from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import ndimage
from numpy import average, absolute
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import sys

LEFT = 0
RIGHT = 1
WIDTH = 1920
HEIGHT = 1080
SAMPLE_LENGTH = 0.2
NUM_CHANNELS = 200
FPS = 60
BACKGROUND_COLOR = 40
BAR_COLOR = (10, 120, 250)
CHANNEL_WIDTH = WIDTH / NUM_CHANNELS
MIN_VISIBLE_HERTZ = 50
MAX_VISIBLE_HERTZ = 40_000
SAMPLE_HERTZ = 1 / SAMPLE_LENGTH

audioFilename = sys.argv[1]
videoFilename = audioFilename.replace(".wav", ".avi")

rate, data = wavfile.read(audioFilename)
MAX_SAMPLE_VALUE = max(abs(data[:,0]))
SECONDS = len(data) / rate
FRAME_COUNT = int(FPS*SECONDS)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter(videoFilename, fourcc, float(FPS), (WIDTH, HEIGHT))
import matplotlib.pyplot as plt

channels = []
channelMultipliers = np.power((np.arange(NUM_CHANNELS) / NUM_CHANNELS), 4) / 2 + 0.5

## initialize channels

for frameIndex in range(FRAME_COUNT):
    time = frameIndex / FPS
    print('creating channel on frame', frameIndex, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
    sampleStartTime = time - (SAMPLE_LENGTH / 2)
    sampleEndTime = time + (SAMPLE_LENGTH / 2)
    sampleStartIndex = max(0, int(sampleStartTime * rate))
    sampleEndIndex = min(int(sampleEndTime * rate), len(data))

    channelsForFrame = np.full(NUM_CHANNELS, 0, dtype=float)
    frequencies = absolute(fft(data[sampleStartIndex:sampleEndIndex,LEFT] / MAX_SAMPLE_VALUE))
    frequencies = frequencies[:len(frequencies) // 2] # ignore negative frequencies

    frequencyPerIndex = len(frequencies) / rate
    lowestAudibleIndex = int(frequencyPerIndex * MIN_VISIBLE_HERTZ)
    highestAudibleIndex = int(frequencyPerIndex * MAX_VISIBLE_HERTZ)
    audibleFrequencies = frequencies[lowestAudibleIndex:highestAudibleIndex]
    audibleFrequencies = audibleFrequencies / len(audibleFrequencies)

    numFrequencies = len(audibleFrequencies)
    logScaleIndices = np.power(numFrequencies, np.arange(1, NUM_CHANNELS + 1) / NUM_CHANNELS) - 1
    sliceRanges = [(int(val), int(logScaleIndices[i + 1] + 1)) for (i, val) in enumerate(logScaleIndices) if i < len(logScaleIndices) - 1]

    for i, (start, end) in enumerate(sliceRanges):
        sliced = audibleFrequencies[start:end]
        channel =  (float(average(sliced)))
        channelsForFrame[i] = channel * channelMultipliers[i]
    channels.append(channelsForFrame)


## process channels

channels = np.sqrt(channels)
channels = ndimage.gaussian_filter(channels, sigma=0.75)

# for i, channelsForFrame in enumerate(channels):
#     time = i / FPS
#     print('processing channel on frame', i, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
#     channels[i] = scipy.signal.convolve(channelsForFrame, [0.1, 0.2, 0.2, 0.4, 0.2, 0.2, 0.1])

## output video

for i, channelsForFrame in enumerate(channels):
    time = i / FPS
    print('creating video on frame', i, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
    for i, channel in enumerate(channelsForFrame):
        cv2.rectangle(frame, (int(i * CHANNEL_WIDTH), HEIGHT), (int((i+1) * CHANNEL_WIDTH), HEIGHT - int(HEIGHT * channel)), BAR_COLOR, int(-1))
    video.write(frame)
video.release()



