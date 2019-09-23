#!/usr/bin/env python

from __future__ import print_function, division
import wave
import numpy as np
import matplotlib.pyplot as plt

# 1번째 방법
def make_graph(img_path,filepath,filename):

    wr = wave.open(filepath, 'r')
    sz = wr.getframerate()
    q = 120  # time window to analyze in seconds
    c = 1  # number of time windows to process
    sf = 0.8  # signal scale factor

    for num in range(c):
        print('Processing from {} to {} s'.format(num*q, (num+1)*q))
        avgf = np.zeros(int(sz/2+1))
        snd = np.array([])
        # The sound signal for q seconds is concatenated. The fft over that
        # period is averaged to average out noise.
        for j in range(q):
            da = np.fromstring(wr.readframes(sz), dtype=np.int16)
            left, right = da[0::2]*sf, da[1::2]*sf
            lf, rf = abs(np.fft.rfft(left)), abs(np.fft.rfft(right))
            snd = np.concatenate((snd, (left+right)/2))
            avgf += (lf+rf)/2
        avgf /= q
        # Plot both the signal and frequencies.
        plt.figure(1)
        a = plt.subplot(211)  # signal
        r = 2**16/2
        a.set_ylim([-r, r])
        a.set_xlabel('time [s]')
        a.set_ylabel('signal [-]')
        x = np.arange(44100*q)/44100
        plt.plot(x, snd)
        b = plt.subplot(212)  # frequencies
        b.set_xscale('log')
        b.set_xlabel('frequency [Hz]')
        b.set_ylabel('|amplitude|')
        plt.plot(abs(avgf))
        plt.savefig(img_path+filename+'.png')
        plt.clf()
# 2
# import matplotlib.pyplot as plt
# import numpy as np
# import wave
# import sys
# from pydub import AudioSegment
#
# sound = AudioSegment.from_wav("q.wav")
# sound = sound.set_channels(1)
# sound.export("out_Car1.wav", format="wav")
#
# spf = wave.open('out_Car1.wav', 'r')
#
# # Extract Raw Audio from Wav File
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, dtype=np.int16)
#
# # If Stereo
# if spf.getnchannels() == 2:
#     print
#     'Just mono files'
#     sys.exit(0)
#
# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(signal)
# plt.show()
