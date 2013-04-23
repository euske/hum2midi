#!/usr/bin/env python
import sys
import midi
from pitch import PitchContour
from wavestream import WaveReader

# MIDI freq/key number.
FREQUENCIES = [
    (28, 21), (29, 22), (31, 23), (33, 24),
    (35, 25), (37, 26), (39, 27), (41, 28),
    (44, 29), (46, 30), (49, 31), (52, 32),
    (55, 33), (58, 34), (62, 35), (65, 36),
    (69, 37), (73, 38), (78, 39), (82, 40),
    (87, 41), (93, 42), (98, 43), (104, 44),
    (110, 45), (117, 46), (123, 47), (131, 48),
    (139, 49), (147, 50), (156, 51), (165, 52),
    (175, 53), (185, 54), (196, 55), (208, 56),
    (220, 57), (233, 58), (247, 59), (262, 60),
    (277, 61), (294, 62), (311, 63), (330, 64),
    (349, 65), (370, 66), (392, 67), (415, 68),
    (440, 69), (466, 70), (494, 71), (523, 72),
    (554, 73), (587, 74), (622, 75), (659, 76),
    (698, 77), (740, 78), (784, 79), (831, 80),
    (880, 81), (932, 82), (988, 83), (1047, 84),
    (1109, 85), (1175, 86), (1245, 87), (1319, 88),
    (1397, 89), (1480, 90), (1568, 91), (1661, 92),
    (1760, 93), (1865, 94), (1976, 95), (2093, 96),
    (2217, 97), (2349, 98), (2489, 99), (2637, 100),
    (2794, 101), (2960, 102), (3136, 103), (3322, 104),
    (3520, 105), (3729, 106), (3951, 107), (4186, 108)
    ]

def getrangelist(pts):
    k0 = None
    r = []
    for (k1,v) in sorted(pts):
        if k0 is None:
            k = -sys.maxint
        else:
            k = (k0+k1)//2
        r.append((k, v))
        k0 = k1
    r.append((sys.maxint, None))
    return r

FRANGES = getrangelist(FREQUENCIES)

def getpt(ranges, p0):
    i0 = 0
    i1 = len(ranges)
    while i0 < i1:
        i = (i0+i1)//2 # i0 <= i <= i1
        (p,_) = ranges[i]
        if p0 < p:
            i1 = i # i0 <= i1
        else:
            i0 = i+1
    return i0

def majority(values):
    c = {}
    m = 0
    for v in values:
        if v not in c: c[v] = 0
        c[v] += 1
        m = max(m, c[v])
    for (v,n) in c.iteritems():
        if n == m: break
    return v

def main(argv):
    import getopt
    from wavestream import WaveReader
    def usage():
        print ('usage: %s [-M|-F] [-n pitchmin] [-m pitchmax] [-t threshold] '
               '[-o out.mid] [-w wsize] [-p instru] wav ...' % argv[0])
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'MFn:m:t:o:w:p:')
    except getopt.GetoptError:
        return usage()
    pitchmin = 70
    pitchmax = 400
    threshold = 0.95
    outpath = 'out.mid'
    wsize = 50
    instru = 0
    attack = 70
    release = 70
    for (k, v) in opts:
        if k == '-M': (pitchmin,pitchmax) = (75,200) # male voice
        elif k == '-F': (pitchmin,pitchmax) = (150,300) # female voice
        elif k == '-n': pitchmin = int(v)
        elif k == '-m': pitchmax = int(v)
        elif k == '-t': threshold = float(v)
        elif k == '-o': outpath = v
        elif k == '-w': wsize = int(v)
        elif k == '-p': instru = int(v)
    contour = None
    for path in args:
        src = WaveReader(path)
        if contour is None:
            contour = PitchContour(src.framerate,
                                   pitchmin=pitchmin, pitchmax=pitchmax,
                                   threshold=threshold)
        contour.load(src.readraw(), src.nframes)
        src.close()
    #
    events = [midi.ProgramChangeEvent(tick=0, channel=0, data=[instru])]
    window = []
    km0 = 0
    kt = 0
    kd = 0
    for p in contour.segments:
        if p == 0:
            k = 0
        else:
            i = getpt(FRANGES, p)
            (_,k) = FRANGES[i-1]
        #print '%d/%d' % (p,k),
        window.append(k)
        if len(window) < wsize: continue
        window = window[1:]
        km1 = majority(window)
        if km0 == km1:
            kd += 1
        else:
            print km0, kd
            if km0 == 0:
                kt += kd
            else:
                events.append(midi.NoteOnEvent(tick=kt, channel=0, data=[km0, attack]))
                events.append(midi.NoteOffEvent(tick=kd, channel=0, data=[km0, release]))
                kt = 0
            kd = 0
            km0 = km1
    events.append(midi.EndOfTrackEvent(tick=0, data=[]))
    pat = midi.Pattern(tracks=[events])
    midi.write_midifile(outpath, pat)
    return

if __name__ == '__main__': sys.exit(main(sys.argv))
