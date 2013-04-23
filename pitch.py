#!/usr/bin/env python
#
# pitch.py
# usage: python pitch.py [-M|-F] [-n pitchmin] [-m pitchmax] wav ...
#

import sys
import wavcorr


##  PitchSegment
##
class PitchSegment(object):

    def __init__(self):
        self.pitches = []
        return

    def __repr__(self):
        (pos1,_) = self.pitches[-1]
        s = '-'.join( '%d-[%d]' % (pos,pitch) for (pos,pitch) in self.pitches[:-1] )
        return '(%s-%d)' % (s, pos1)

    def add(self, pos, pitch):
        self.pitches.append((pos, pitch))
        return

    def finish(self):
        self.pos0 = min( pos for (pos,_) in self.pitches )
        self.pos1 = max( pos for (pos,_) in self.pitches )
        x = 0
        for i in xrange(len(self.pitches)-1):
            (pos0,pitch0) = self.pitches[i]
            #print pos0, pitch0
            (pos1,pitch1) = self.pitches[i+1]
            x += (pitch0+pitch1)*(pos1-pos0)/2
        self.avg = x/(self.pos1-self.pos0)
        return

    def getsrc(self, pos):
        for i in xrange(len(self.pitches)-1):
            (pos0,pitch0) = self.pitches[i]
            (pos1,pitch1) = self.pitches[i+1]
            if pos0 <= pos and pos <= pos1:
                return (pos-pos0)*(pitch1-pitch0)/(pos1-pos0)+pitch0
        return 0

    def getavg(self, pos):
        return self.avg


##  PitchContour
##
class PitchContour(object):

    def __init__(self, framerate,
                 pitchmin=70, pitchmax=400, threshold=0.7):
        self.framerate = framerate
        self.wmin = (framerate/pitchmax)
        self.wmax = (framerate/pitchmin)
        self.threshold = threshold
        self.reset()
        return

    def reset(self):
        self.segments = []
        self._offset = 0
        return
    
    def load(self, buf, nframes):
        #print 'detection: %s samples...' % len(wav)
        i = 0
        while i+self.wmax < nframes:
            (dmax, mmax) = wavcorr.autocorrs16(self.wmin, self.wmax, buf, i)
            t = self._offset + i
            if self.threshold < mmax:
                pitch = self.framerate/dmax
                self.segments.append((t, pitch))
            elif mmax < 0.5:
                self.segments.append((t, 0))
            i += self.wmin/2
        self._offset += nframes
        return

# main
def main(argv):
    import getopt
    from wavestream import WaveReader
    def usage():
        print 'usage: %s [-M|-F] [-n pitchmin] [-m pitchmax] [-t threshold] wav ...' % argv[0]
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'MFn:m:t:')
    except getopt.GetoptError:
        return usage()
    pitchmin = 70
    pitchmax = 400
    threshold = 0.9
    for (k, v) in opts:
        if k == '-M': (pitchmin,pitchmax) = (75,200) # male voice
        elif k == '-F': (pitchmin,pitchmax) = (150,300) # female voice
        elif k == '-n': pitchmin = int(v)
        elif k == '-m': pitchmax = int(v)
        elif k == '-t': threshold = float(v)
    contour = None
    for path in args:
        src = WaveReader(path)
        if contour is None:
            contour = PitchContour(src.framerate,
                                   pitchmin=pitchmin, pitchmax=pitchmax,
                                   threshold=threshold)
        contour.load(src.readraw(), src.nframes)
        src.close()
    for (t,p) in contour.segments:
        print t, p
    return

if __name__ == '__main__': sys.exit(main(sys.argv))
