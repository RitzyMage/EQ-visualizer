from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import ndimage
from numpy import average, absolute
import numpy as np
import math
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

LEFT = 0
RIGHT = 1
WIDTH = 1920
HEIGHT = 1080
SAMPLE_LENGTH = 0.1
NUM_CHANNELS = 40
FPS = 24
BACKGROUND_COLOR = 40
BAR_COLOR = (10, 120, 250)
CHANNEL_WIDTH = WIDTH / NUM_CHANNELS
MIN_VISIBLE_HERTZ = 100
MAX_VISIBLE_HERTZ = 20_000
SAMPLE_HERTZ = 1 / SAMPLE_LENGTH
FREQUENCY_LOG_BASE = 1.005 # the EQ should not be linear, this is the base of the exponent we're using. should be >1

rate, data = wavfile.read('SineC_EQ.wav')
MAX_SAMPLE_VALUE = max(abs(data[:,0]))
SECONDS = len(data) / rate
FRAME_COUNT = int(FPS*SECONDS)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./noise.avi', fourcc, float(FPS), (WIDTH, HEIGHT))
import matplotlib.pyplot as plt

channels = []

## initialize channels

for frameIndex in [20]:#range(FRAME_COUNT):
    time = frameIndex / FPS
    print('creating channel on frame', frameIndex, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
    sampleStartTime = time - (SAMPLE_LENGTH / 2)
    sampleEndTime = time + (SAMPLE_LENGTH / 2)
    sampleStartIndex = max(0, int(sampleStartTime * rate))
    sampleEndIndex = min(int(sampleEndTime * rate), len(data))

    frameData = data[sampleStartIndex:sampleEndIndex,LEFT] / MAX_SAMPLE_VALUE
    frameFourier = absolute(fft(frameData))
    frequencyPerIndex = rate / len(frameData)
    frameFourier = frameFourier[:len(frameFourier) // 2]
    lowestAudibleIndex = int(MIN_VISIBLE_HERTZ / frequencyPerIndex)
    highestAudibleIndex = int(MAX_VISIBLE_HERTZ / frequencyPerIndex)
    print(len(frameFourier), rate, frequencyPerIndex, lowestAudibleIndex, highestAudibleIndex)
    frameFourier = frameFourier[lowestAudibleIndex:highestAudibleIndex]

    fig, ax = plt.subplots()

    ax.plot(np.log(range(len(frameFourier))), frameFourier, linewidth=2.0)

    plt.show()

    channelsForFrame = np.full(NUM_CHANNELS, 0, dtype=float)
    frequencies = absolute(fft(data[sampleStartIndex:sampleEndIndex,LEFT] / MAX_SAMPLE_VALUE))
    frequencies = frequencies[:len(frequencies) // 2] # ignore negative frequencies

    frequencyPerIndex = len(frequencies) / rate
    lowestAudibleIndex = int(frequencyPerIndex * MIN_VISIBLE_HERTZ)
    highestAudibleIndex = int(frequencyPerIndex * MAX_VISIBLE_HERTZ)
    audibleFrequencies = frequencies[lowestAudibleIndex:highestAudibleIndex]


    numFrequencies = len(audibleFrequencies)
    firstSliceSize = numFrequencies / ((FREQUENCY_LOG_BASE ** (NUM_CHANNELS) - 1) / (FREQUENCY_LOG_BASE - 1)) #did some math to figure this out
    lastSliceSize = firstSliceSize
    sliceRanges = [(numFrequencies - lastSliceSize, numFrequencies)]

    for _ in range(NUM_CHANNELS - 1):
        lastSliceSize = lastSliceSize * FREQUENCY_LOG_BASE
        sliceEnd = sliceRanges[len(sliceRanges) - 1][0]
        sliceRanges.append((sliceEnd - lastSliceSize, sliceEnd))
    sliceRanges = [(int(start), math.ceil(end)) for start, end in sliceRanges]


    for i, (start, end) in enumerate(sliceRanges[::-1]):
        sliced = audibleFrequencies[start:end]
        channel =  (float(average(sliced)))
        channelsForFrame[i] = channel
    channels.append(channelsForFrame)

## process channels

#channels = ndimage.gaussian_filter(channels, sigma= 4)

# for i, channelsForFrame in enumerate(channels):
#     time = i / FPS
#     print('processing channel on frame', i, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
#     channels[i] = scipy.signal.convolve(channelsForFrame, [0.1, 0.2, 0.2, 0.4, 0.2, 0.2, 0.1])

## output video

# for i, channelsForFrame in enumerate(channels):
#     time = i / FPS
#     print('creating video on frame', i, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
#     frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
#     for i, channel in enumerate(channelsForFrame):
#         cv2.rectangle(frame, (int(i * CHANNEL_WIDTH), HEIGHT), (int((i+1) * CHANNEL_WIDTH), HEIGHT - int(HEIGHT * channel)), BAR_COLOR, int(-1))
#     video.write(frame)
# video.release()



