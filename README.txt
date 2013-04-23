hum2midi.py

Create a MIDI file from a hum by using pitch detection.

Thanks to:
  python-midi: https://github.com/vishnubob/python-midi/


Usage:
  $ make
    (C compile is needed.)
  $ python hum2midi.py [-M|-F] -o out.mid input.wav

Options:
  -M : Specifies a male voice as input.
  -F : Specifies a female voice as input.
