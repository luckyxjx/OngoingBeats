import pretty_midi
import glob

files = glob.glob('generated_api/*.mid')
if files:
    f = files[0]
    m = pretty_midi.PrettyMIDI(f)
    print(f'File: {f}')
    print(f'Instruments: {len(m.instruments)}')
    print(f'Notes: {sum(len(i.notes) for i in m.instruments)}')
    print(f'Duration: {m.get_end_time():.2f}s')
else:
    print('No MIDI files found')
