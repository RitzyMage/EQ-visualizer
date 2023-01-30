from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import ndimage
from scipy import interpolate
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
NUM_CHANNELS = 100
FPS = 24
BACKGROUND_COLOR = 30
SPECTRUM_COLOR_1 = (102, 94, 8)
SPECTRUM_COLOR_2 = (139, 161, 43)
SPECTRUM_COLOR_3 = (107, 190, 50)
CHANNEL_WIDTH = WIDTH / NUM_CHANNELS
MIN_VISIBLE_HERTZ = 100
MAX_VISIBLE_HERTZ = 25_000
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
channelMultipliers = 0.95 * (np.arange(NUM_CHANNELS) / NUM_CHANNELS) + 0.05

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
        channel =  (float(max(sliced)))
        channelsForFrame[i] = channel * channelMultipliers[i]
    channels.append(channelsForFrame)


## process channels

channelsBackground = ndimage.gaussian_filter(np.power(channels, 0.4) , sigma=1.25)
channelsMidground = ndimage.gaussian_filter(np.power(channels, 0.55) , sigma=1)
channelsForeground = ndimage.gaussian_filter(np.power(channels, 0.9) , sigma=0.75)

## output video

def writeFrameFromChannels(frame, channelsForFrame, color):
    splineInterpolation = interpolate.CubicSpline(np.arange(len(channelsForFrame)) * CHANNEL_WIDTH, channelsForFrame)
    interpolatedPoints = HEIGHT * (1 - splineInterpolation(np.arange(WIDTH)))
    eqPoints = list(enumerate(interpolatedPoints))
    cv2.fillPoly(frame, [np.array([[int(0), int(HEIGHT)]] + eqPoints +  [[WIDTH, HEIGHT]], dtype=np.int32)], color, lineType=cv2.LINE_AA)

for i in range(len(channels)):
    time = i / FPS
    print('creating video on frame', i, 'time:\t', int(time // 60), '\t:\t', int(time % 60))
    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    writeFrameFromChannels(frame, channelsBackground[i], SPECTRUM_COLOR_1)
    writeFrameFromChannels(frame, channelsMidground[i], SPECTRUM_COLOR_2)
    writeFrameFromChannels(frame, channelsForeground[i], SPECTRUM_COLOR_3)
    
    video.write(frame)
video.release()



