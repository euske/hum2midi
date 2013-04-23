# Makefile

RM=rm -f
CP=cp -f
PYTHON=python
HUM2MIDI=$(PYTHON) hum2midi.py

all: wavcorr.so

clean:
	-$(RM) -r build
	-$(RM) *.pyc *.pyo
	-$(RM) midi/*.pyc midi/*.pyo
	-$(RM) wavcorr.so
	-$(RM) mary.mid

wavcorr.so: wavcorr.c
	$(PYTHON) setup.py build
	$(CP) build/lib.*/wavcorr.so .

pitch.py: wavcorr.so

hum2midi.py: pitch.py wavestream.py

test: mary.mid

mary.mid: hum2midi.py mary.wav
	$(HUM2MIDI) -M -o $@ mary.wav
