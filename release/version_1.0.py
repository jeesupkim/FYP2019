import mido
import numpy as np
import itertools
import random

MAX_NUM_OF_CHANNELS = 16
DRUM_CHANNEL = 9
MOODS_NAME = ['angry', 'calm', 'happy', 'sad']
MOOD_ANGRY = 0
MOOD_CALM = 1
MOOD_HAPPY = 2
MOOD_SAD = 3

# return true if the message is a note
def is_note(msg):
    return msg.type == 'note_on' or msg.type == 'note_off'

# return true if the message indicates the start of a note
# return false if the message indicates the end of a note
def is_note_on(msg):
    return msg.type == 'note_on' and msg.velocity != 0

def is_drum(msg):
    return msg.channel == DRUM_CHANNEL

# get the time and tick indicating the bars, and their beat and tempo
def get_bars_info(midi_file):
    
    ticks_per_beat = midi_file.ticks_per_beat

    total_tick = 0
    total_time = 0
    rhythm_events = []

    # store the time signature and tempo change events as rhythm_events
    for msg in midi_file:

        # sum up the tick/time
        if msg.time != 0:
            total_tick += mido.second2tick(msg.time, ticks_per_beat, current_tempo)
            total_time += msg.time

        # store meta message of time signature and tempo with current tick
        if msg.is_meta:

            if msg.type == 'time_signature':
                time_signature_quarter = msg.numerator / (msg.denominator / 4)
                rhythm_events.append([total_tick, -1, time_signature_quarter])

            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                rhythm_events.append([total_tick, current_tempo, -1])
    
    # store the tick of the end of midi
    rhythm_events.append([total_tick, -1, -1])

    cumulative_time = 0
    cumulative_tick = 0
    prev_tick = 0
    current_tempo = -1
    current_ts_quarter = -1
    bars_time = []
    bars_tick = []
    bars_beat = []
    bars_tempo = []

    # append the time/tick at each bar according to 'rhythm_events'
    for rhythm_event in rhythm_events:

        # detect rhythm change, then conclude the bars before the change
        if rhythm_event[0] != prev_tick:

            tick_diff = rhythm_event[0] - prev_tick
            prev_tick = rhythm_event[0]

            # conclude the bar time/tick by dividing the bar length
            # given total tick, and constant beat and tempo, the bars can be separated by dividing evenly
            num_of_bar = tick_diff / ticks_per_beat
            num_of_bar_2 = num_of_bar
            while num_of_bar > 0:
                bar_tick = (num_of_bar_2 - num_of_bar) * ticks_per_beat
                bars_time.append(mido.tick2second(bar_tick, ticks_per_beat, current_tempo) + cumulative_time)
                bars_tick.append(bar_tick + cumulative_tick)
                bars_beat.append(current_ts_quarter)
                bars_tempo.append(current_tempo)
                num_of_bar -= current_ts_quarter            

            cumulative_time += mido.tick2second(tick_diff, ticks_per_beat, current_tempo)
            cumulative_tick += tick_diff

        # update the time signature and time tempo events
        if rhythm_event[1] != -1:
            current_tempo = rhythm_event[1]
        if rhythm_event[2] != -1:
            current_ts_quarter = rhythm_event[2]

    return bars_time, bars_tick, bars_beat, bars_tempo

# get the details of notes in each bar
# i.e. bars_note_info[bar index] = [start tick of the bar, end tick of the bar, bar_note_info]
# i.e. bar_note_info[note index] = [channel index, note no., triggered time (in tick), 1 = 'note_on' / 0 = 'note_off']
def get_bars_note_info(midi_file):

    _, bars_tick, bars_beat, _ = get_bars_info(midi_file)

    ticks_per_beat = midi_file.ticks_per_beat
    current_tempo = -1
    current_ts_quarter = -1

    total_tick = 0
    bar_start_tick = bars_tick.pop(0)
    bar_note_info = []
    bars_note_info = []

    # store the time signature and tempo change events as rhythm_events
    for msg in midi_file:

        # sum up the tick/time
        if msg.time != 0:        
            total_tick += mido.second2tick(msg.time, ticks_per_beat, current_tempo)
            if len(bars_tick) != 0 and total_tick >= bars_tick[0]:

                # find the suitable bar for placing current note
                bars_shift_idx = 0
                while len(bars_tick) > (bars_shift_idx + 1) and total_tick >= bars_tick[bars_shift_idx + 1]:
                    bars_shift_idx += 1

                bars_note_info.append([bar_start_tick, bars_tick[0] - bar_start_tick, bar_note_info, bars_beat[0]])
                bar_note_info = []
                bar_start_tick = bars_tick.pop(0)
                bars_beat.pop(0)

                # put back the shifted (empty) bars
                for pad_bar_idx in range(bars_shift_idx):
                    bars_note_info.append([bar_start_tick, bars_tick[0] - bar_start_tick, bar_note_info, bars_beat[0]])
                    bar_note_info = []
                    bar_start_tick = bars_tick.pop(0)
                    bars_beat.pop(0)


        # store meta message of time signature and tempo with current tick
        if msg.is_meta:

            if msg.type == 'time_signature':
                current_ts_quarter = msg.numerator / (msg.denominator / 4)

            if msg.type == 'set_tempo':
                current_tempo = msg.tempo

        if not msg.is_meta:

            if is_note(msg):

                if is_note_on(msg):
                    bar_note_info.append([msg.channel, msg.note, total_tick, 1, msg.velocity])

                else:
                    bar_note_info.append([msg.channel, msg.note, total_tick, 0, msg.velocity])
    return bars_note_info

# get rhythm pattern in each bar of each channel
# i.e. tracks_rhythm_pattern[channel index, bar index] = [[note1 start_time, note1 duration], [note2 start_time, note2 duration], ...]
def get_rhythm_pattern(bars_note_info):

    MAX_NUM_OF_CHANNELS = 16
    MAX_NUM_OF_NOTE_IN_BAR = 64
    tracks_rhythm_pattern = np.zeros([MAX_NUM_OF_CHANNELS, len(bars_note_info), MAX_NUM_OF_NOTE_IN_BAR, 5]) - 1
    # print(tracks_rhythm_pattern.shape)


    prev_bar_notes = []

    for idx, bar_note_info in enumerate(bars_note_info):

        if len(prev_bar_notes) != 0:
            for prev_bar_note in prev_bar_notes:
                append_idx = 0
                while append_idx < 64:
                    if tracks_rhythm_pattern[prev_bar_note[0], idx, append_idx, 0] == -1:
                        tracks_rhythm_pattern[prev_bar_note[0], idx, append_idx] = [0, prev_bar_note[1], 0, 0, 0]
                        break
                    append_idx += 1
            prev_bar_notes = []

        # print(idx, bar_note_info[2])
        for note in bar_note_info[2]:
            if note[3]:
                append_idx = 0
                while append_idx < 64:
                    if tracks_rhythm_pattern[note[0], idx, append_idx, 0] == -1:
                        tracks_rhythm_pattern[note[0], idx, append_idx, 0] = note[2] - bar_note_info[0]
                        tracks_rhythm_pattern[note[0], idx, append_idx, 1] = -2 - note[1]
                        break
                    append_idx += 1
            else:
                # print(tracks_rhythm_pattern[note[0], idx, :, 1], -2 - note[1])
                relative_idx = np.nonzero(tracks_rhythm_pattern[note[0], idx, :, 1] == -2 - note[1])[0][0]
                tracks_rhythm_pattern[note[0], idx, relative_idx, 1] = note[2] - bar_note_info[0] - tracks_rhythm_pattern[note[0], idx, relative_idx, 0]
                # print(tracks_rhythm_pattern[note[0], idx, relative_idx])

        for channel_idx in range(MAX_NUM_OF_CHANNELS):

            temp_idx = 0

            while temp_idx < MAX_NUM_OF_NOTE_IN_BAR and tracks_rhythm_pattern[channel_idx, idx, temp_idx, 1] != -1:
                if tracks_rhythm_pattern[channel_idx, idx, temp_idx, 1] < -1:
                    prev_bar_notes.append([channel_idx, tracks_rhythm_pattern[channel_idx, idx, temp_idx, 1]])
                    tracks_rhythm_pattern[channel_idx, idx, temp_idx, 1] = bar_note_info[1] - tracks_rhythm_pattern[note[0], idx, relative_idx, 0]
                    # print(tracks_rhythm_pattern[channel_idx, idx, temp_idx])
                tracks_rhythm_pattern[channel_idx, idx, temp_idx] /= bar_note_info[1]
                temp_idx += 1
        # print(prev_bar_notes)
    
    return tracks_rhythm_pattern

def get_midi_info(bars_note_info):
    midi_info = np.empty((len(bars_note_info), MAX_NUM_OF_CHANNELS), dtype=np.ndarray)
    for i in range(midi_info.shape[0]):
        for j in range(midi_info.shape[1]):
            midi_info[i, j] = np.empty((0, 7))


    prev_bar_notes = []

    for idx, bar_note_info in enumerate(bars_note_info):

        if len(prev_bar_notes) != 0:
            for prev_bar_note in prev_bar_notes:
                midi_info[idx, prev_bar_note[0]] = np.vstack((midi_info[idx, prev_bar_note[0]], [prev_bar_note[2], prev_bar_note[3], 0, prev_bar_note[1], 0, 0, 1]))
            prev_bar_notes = []

        for note in bar_note_info[2]:
            if note[3]:
                midi_info[idx, note[0]] = np.vstack((midi_info[idx, note[0]], [note[1], note[4], note[2] - bar_note_info[0], -2 - note[1], 0, 0, 0]))
            else:
                relative_idx = np.nonzero(midi_info[idx, note[0]][:, 3] == -2 - note[1])[0][0]
                midi_info[idx, note[0]][relative_idx, 3] = note[2] - bar_note_info[0] - midi_info[idx, note[0]][relative_idx, 2]

        for channel_idx in range(MAX_NUM_OF_CHANNELS):

            for i, info in enumerate(midi_info[idx, channel_idx]):
                if info[3] < -1:
                    prev_bar_notes.append([channel_idx, info[3], info[0], info[1]])
                    midi_info[idx, channel_idx][i, 3] = bar_note_info[1] - info[2]
                    midi_info[idx, channel_idx][i, 6] = 1
            midi_info[idx, channel_idx][:, 2:4] /= bar_note_info[1]
            midi_info[idx, channel_idx][:, 4:6] = midi_info[idx, channel_idx][:, 2:4] * bar_note_info[3]
    return midi_info

# turn numeric notes into english notes, octave info not preserved
def note_int2char(note):
    return {
        '0': 'C', '1': 'Db', '2': 'D', '3': 'Eb',
        '4': 'E', '5': 'F', '6': 'F#', '7': 'G',
        '8': 'Ab', '9': 'A', '10': 'Bb', '11': 'B'
    }.get(str(note % 12), 'undef')

# define the chords and its relative position
chords = np.empty((0, 12), dtype=np.int8)

chords_entries = np.array([
    # C1
    [6, -1, 0, -4, -4, 0, -1, -5, -1, -1, -3, -3],
    # C5
    [5, -1, 0, -4, -4, 0, -1, 5, -1, -1, -3, -3],
    # C
    [5, -3, 0, -4, 4, 0, -3, 4, -3, 1, -4, -4],
    # Cm
    [5, -3, 0, 4, -4, 0, -3, 4, 1, 0, -4, -4],
    # C7
    [5, -3, 0, -3, 3, 0, -3, 3, 1, 1, 3, -4],
    # CMaj7
    [5, -3, 0, -3, 3, 0, -3, 3, -3, 1, -3, 3],
    # Cmin7
    [5, -3, 0, 3, -3, 0, -3, 3, 1, 0, 3, -1],
    # Cdim
    [5, 0, -3, 4, -3, -3, 4, -4, 0, -3, -3, -3],
], dtype=np.int8)

for chords_entry in chords_entries:
    for i in range(12):
        chords = np.vstack((chords, np.roll(chords_entry, i)))

# name the chord
def name_chord(chord):
    return {
        '0': note_int2char(chord) + '1',
        '1': note_int2char(chord) + '5',
        '2': note_int2char(chord),
        '3': note_int2char(chord) + 'm',
        '4': note_int2char(chord) + '7',
        '5': note_int2char(chord) + 'M7',
        '6': note_int2char(chord) + 'm7',
        '7': note_int2char(chord) + 'dim'
    }.get(str(chord // 12), 'undef')

# chords to key, '[C, Db, D, ..., B, Cm, Dbm, ..., Bbm, Bm]'
chord_to_key = np.empty((0, 36), dtype=np.int8)

chord_to_key_entries = np.array([
    # C note only
    [2, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    # C power chords
    [2, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    # C Major chords
    [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    # C Minor chords
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    # C Dominant 7th chords
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # C Major 7th chords
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # C Minor 7th chords
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # C Diminished chords
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
], dtype=np.int8)

for chord_to_key_entry in chord_to_key_entries:
    for i in range(12):
        chord_to_key = np.vstack((chord_to_key, np.concatenate((np.roll(chord_to_key_entry[0:12], i), np.roll(chord_to_key_entry[12:24], i), np.roll(chord_to_key_entry[24:36], i)))))

# name the key
def name_key(key):
    return {
        '0': note_int2char(key) + ' Major',
        '1': note_int2char(key) + ' minor',
        '2': note_int2char(key) + ' minor (harmonic)',
    }.get(str(key // 12), 'undef')

# get chords of a key
def key_to_chords(key):
    return np.nonzero(np.transpose(chord_to_key)[key])

def get_chords(midi_info):
    chords_predict = np.zeros(midi_info.shape[0], dtype=np.uint8)
    for bar_idx in range(midi_info.shape[0]):
        notes_stat = np.zeros(12, dtype=np.float)
        for channel_idx in range(MAX_NUM_OF_CHANNELS):
            if channel_idx != DRUM_CHANNEL:
                for info in midi_info[bar_idx, channel_idx]:
                    notes_stat[int(info[0] % 12)] += info[3]
        chords_score = np.dot(chords, notes_stat)
        chords_predict[bar_idx] = np.argmax(chords_score)
#         with np.printoptions(precision=3, suppress=True):
#             print(notes_stat)
#             print(["%s: %f"%(note_int2char(idx), note) for idx, note in enumerate(notes_stat) if note > 0])
#             print(name_chord(chords_predict[bar_idx]))
#             print([name_chord(chord) for chord in np.argsort(chords_score)[-10:]])
#             print(chords_score[np.argsort(chords_score)[-10:]])
    return chords_predict

def get_key(chords_predict):
    chords_stat = np.zeros(chords.shape[0], dtype=np.int)
    for chord_predict in chords_predict:
        chords_stat[chord_predict] += 1
    key_stat = np.dot(chords_stat, chord_to_key)
    key_predict = np.argmax(key_stat)
#     with np.printoptions(precision=3, suppress=True):
#         print([name_key(key) for key in np.argsort(key_stat)[-3:]])
#         print(key_stat[np.argsort(key_stat)[-3:]])
    key_acc = np.sum(chords_stat[key_to_chords(key_predict)]) / np.sum(chords_stat)
    return key_predict, key_acc, chords_stat

# keys notes, '[C, Db, D, ..., B, Cm, Dbm, ..., Bbm, Bm]'
keys_notes = np.empty((0, 12), dtype=np.int8)

keys_notes_entries = np.array([
    # C Major
    [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # C minor
    [2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
], dtype=np.int8)

for keys_notes_entry in keys_notes_entries:
    for i in range(12):
        keys_notes = np.vstack((keys_notes, np.roll(keys_notes_entry[0:12], i)))

# chords notes, '[C, Db, D, ..., B, Cm, Dbm, ..., Bbm, Bm]'
chords_notes = np.empty((0, 12), dtype=np.int8)

chords_notes_entries = np.array([
    # C1
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C5
    [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # C
    [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # Cm
    [2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    # C7
    [2, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    # CMaj7
    [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # Cmin7
    [2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    # Cdim
    [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=np.int8)

for chords_notes_entry in chords_notes_entries:
    for i in range(12):
        chords_notes = np.vstack((chords_notes, np.roll(chords_notes_entry, i)))
        
# chords element, '[C, Db, D, ..., B, Cm, Dbm, ..., Bbm, Bm]'
chords_elements = np.empty((0, 12), dtype=np.int8)

chords_elements_entries = np.array([
    # C1
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C5
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # C
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    # Cm
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    # C7
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    # CMaj7
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    # Cmin7
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    # Cdim
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=np.int8)

for chords_elements_entry in chords_elements_entries:
    for i in range(12):
        chords_elements = np.vstack((chords_elements, np.roll(chords_elements_entry, i)))

def is_note_in_chord(note, chord):
    return chords_elements[(int)(chord)][(int)(note) % 12] == 1        

# get each note scores in a bar based on only its note (higher score = more related to the key)
def get_bar_notes_priority(bar_info, key_predict, chord_predict):
    notes_score = keys_notes[key_predict] + chords_notes[chord_predict]
    bar_notes_priority = np.zeros(bar_info.shape[0], dtype=np.uint8)
    for idx, note_info in enumerate(bar_info):
        bar_notes_priority[idx] = notes_score[(int)(note_info[0] % 12)]
    return bar_notes_priority

def get_msg_non_note(midi_file):
    msg_non_note = np.empty((0,2))
    total_tick = 0
    ticks_per_beat = midi_file.ticks_per_beat
    current_tempo = 0

    for msg in midi_file:
        if msg.time != 0:
            total_tick += mido.second2tick(msg.time, ticks_per_beat, current_tempo)

        if msg.is_meta:
            # update tempo for 'second2tick'
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
            msg_non_note = np.vstack((msg_non_note, np.array([total_tick, msg])))
        else:
            if not is_note(msg):
                msg_non_note = np.vstack((msg_non_note, np.array([total_tick, msg])))
    return msg_non_note

def mood_change(nMood, midi_info, midi_chord):
    for bar in range(midi_info.shape[0]):
        for channel in range(midi_info.shape[1]):
            if nMood == MOOD_ANGRY: #angry
                if channel != DRUM_CHANNEL:
                    midi_info[bar, channel] = thicker(midi_info, midi_chord, bar, channel)
                    midi_info[bar, channel] = moreNote("high", midi_info, bar, channel)
            if nMood == MOOD_CALM: #calm
                midi_info[bar, channel] = lessNote("high", midi_info, midi_chord, bar, channel)
                midi_info[bar, channel] = lessSyn(midi_info, bar, channel)
                midi_info[bar, channel] = thinner(midi_info, bar, channel)
            if nMood == MOOD_HAPPY and channel != DRUM_CHANNEL: #happy
                if channel != DRUM_CHANNEL:
                    midi_info[bar, channel] = thicker(midi_info, midi_chord, bar, channel)
                    midi_info[bar, channel] = moreNote("low", midi_info, bar, channel)
                midi_info[bar, channel] = moreSyn("high", midi_info, bar, channel)
            if nMood == MOOD_SAD: #sad
                midi_info[bar, channel] = lessNote("low", midi_info, midi_chord, bar, channel)
                midi_info[bar, channel] = lessSyn(midi_info, bar, channel)

def check_on_beat(degree, start_time, no_of_beat): #checked
    each_beat = 1 / no_of_beat
    on_beat = np.array([0])
    for a in range(1, no_of_beat):
        np.append(on_beat, a * each_beat)
        
    for b in on_beat:
        if ((abs(start_time - b) <= 0.15) and degree == "low") or ((abs(start_time - b) <= 0.35) and degree == "high") :
            return True
    return False

def moreNote(degree, midi_info, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    for note in range(bar_channel_info.shape[0]):
        note_info = bar_channel_info[note]
        mod3 = random.randint(0, 2)
        mod5 = random.randint(0, 4)
        if ((note + 1) % 3 == mod3 and degree == "high") or ((note + 1) % 5 == mod5 and degree == "low"):
            note_info[5] /= 2
            note_info[3] /= 2
            newNote = np.array([-1, note_info[1], note_info[2] + note_info[3], note_info[3], note_info[4] + note_info[5], note_info[5], 0])
            bar_channel_info = np.insert(bar_channel_info, note + 1, newNote, 0)
    return bar_channel_info

def lessNote(degree, midi_info, midi_chord, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    note_priority = get_bar_notes_priority(bar_channel_info, key_predict, midi_chord[bar])
    priority_sort = np.argsort(note_priority)
    cut_quota = (int)(round(bar_channel_info.shape[0] * 0.4))
    cut_list = np.empty(1)
    for note in reversed(range(bar_channel_info.shape[0])):
        note_info = bar_channel_info[note]

        if note_info[3] != 0 and not priority_sort is None:
            if not check_on_beat("low", note_info[2], (int)(round(note_info[5] / note_info[3]))) and (np.where(priority_sort == note)[0] <= cut_quota):
                if note - 1 > 0:
                    dur = note_info[5]
                    wDur = round(bar_channel_info[note - 1][5] / bar_channel_info[note - 1][3])
                    bar_channel_info[note - 1][5] += dur
                    bar_channel_info[note - 1][3] = bar_channel_info[note - 1][5] / wDur
                    cut_list = np.append(cut_list, note)
    cut_list = np.delete(cut_list, 0, 0)
    cut_list = np.flip(cut_list)
    for a in range(cut_list.shape[0]):
        for b in range(cut_list.shape[0]):
            if b < a and cut_list[b] < cut_list[a]:
                cut_list[a] -= 1
    for c in range(cut_list.shape[0]):
        if bar_channel_info[c, 6] == 0:
            bar_channel_info = np.delete(bar_channel_info, c, 0)
    return bar_channel_info

def moreSyn(degree, midi_info, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    for note in range(bar_channel_info.shape[0]):
        note_info = bar_channel_info[note]
        if note_info[3] != 0 and note_info[6] == 0:
            if check_on_beat(degree, note_info[2], int(round(note_info[5] / note_info[3]))):
                at1 = bar_channel_info[note][3] / 2
                at2 = bar_channel_info[note][5] / 2
                bar_channel_info[note][2] += at1
                bar_channel_info[note][3] = at1
                bar_channel_info[note][4] += at2
                bar_channel_info[note][5] = at2
    return bar_channel_info
    
def lessSyn(midi_info, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    if bar_channel_info.shape[0] > 0:
        cut_count = 0.3 * bar_channel_info.shape[0]
        t = 0
        for t in range(bar_channel_info.shape[0]):
            if bar_channel_info[t][3] == 0:
                break
            no_of_beat = int(round(bar_channel_info[t][5] / bar_channel_info[t][3]))
            cut_list = np.array([1, 1.5, 2, 3, 4])
            l0 = np.zeros(1)
            l1 = np.zeros(1)
            l2 = np.zeros(1)
            l3 = np.zeros(1)
            l4 = np.zeros(1)

            for a in range(cut_list.shape[0]):
                temp = 1 / (no_of_beat * cut_list[a])
                temp_2 = temp
                while temp_2 < 0.9999:
                    if a == 0:
                        l0 = np.append(l0, temp_2)
                    if a == 1:
                        l1 = np.append(l1, temp_2)
                    if a == 2:
                        l2 = np.append(l2, temp_2)
                    if a == 3:
                        l3 = np.append(l3, temp_2)
                    if a == 4:
                        l4 = np.append(l4, temp_2)
                    temp_2 += temp

            priority_list = np.zeros(bar_channel_info.shape[0])

            for b in range(priority_list.shape[0]):
                for c in range(l0.shape[0]):
                    if abs(bar_channel_info[b][2] - l0[c]) < 0.05:
                        priority_list[b] = 5
                for d in range(l1.shape[0]):
                    if abs(bar_channel_info[b][2] - l1[d]) < 0.05:
                        priority_list[b] = 4
                for e in range(l2.shape[0]):
                    if abs(bar_channel_info[b][2] - l2[e]) < 0.05:
                        priority_list[b] = 3
                for f in range(l3.shape[0]):
                    if abs(bar_channel_info[b][2] - l3[f]) < 0.05:
                        priority_list[b] = 2
                for g in range(l4.shape[0]):
                    if abs(bar_channel_info[b][2] - l4[c]) < 0.05:
                        priority_list[b] = 1

            sorted_pri = np.argsort(priority_list)
            note_idx = np.empty(0)
            tar_val = np.empty(0)
            for h in range(sorted_pri.shape[0]):
                if h < cut_count:
                    for i in range(l0.shape[0]):
                        if bar_channel_info[sorted_pri[h]][2] < l0[i]:
                            note_idx = np.append(note_idx, sorted_pri[h])
                            tar_val = np.append(tar_val, l0[i - 1])
                            break
                        if bar_channel_info[sorted_pri[h]][2] > l0[-1]:
                            note_idx = np.append(note_idx, sorted_pri[h])
                            tar_val = np.append(tar_val, l0[-1])
                            break
            if note_idx.shape[0] > 0:
                for j in range(note_idx.shape[0]):
                    bar_channel_info[(int)(note_idx[j])][2] = tar_val[j]
    return bar_channel_info
    
def thicker(midi_info, midi_chord, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    if bar_channel_info.shape[0] > 0:
        bar_chord = midi_chord[bar]
        newNote = np.zeros(1)
        flag = 0
        for note in range(bar_channel_info.shape[0]):
            note_info = bar_channel_info[note]
            if is_note_in_chord(note_info[0], bar_chord):
                for a in range(1, 12):
                    if note_info[0] > 12 and is_note_in_chord(note_info[0] - a, bar_chord):
                        newNote = np.array([note_info[0] - a, note_info[1], note_info[2], note_info[3], note_info[4], note_info[5], 0])
                        flag = 1
        if flag == 1:
            bar_channel_info = np.insert(bar_channel_info, note + 1, newNote, 0)
    return bar_channel_info

def thinner(midi_info, bar, channel): #checked
    bar_channel_info = midi_info[bar, channel]
    cut_list = np.empty(0)
    for note1 in range(bar_channel_info.shape[0]):

        for note2 in range(bar_channel_info.shape[0]):
            if note2 > note1 and bar_channel_info[note2][2] == bar_channel_info[note1][2] and (not note2 in cut_list) and bar_channel_info[note2, 6] == 0:
                cut_list = np.append(cut_list, note2)
    for a in range(cut_list.shape[0]):
        for b in range(cut_list.shape[0]):
            if b > a and cut_list[a] < cut_list[b]:
                cut_list[b] -= 1
    for c in range(cut_list.shape[0]):
        bar_channel_info = np.delete(bar_channel_info, cut_list[c], 0)
    return bar_channel_info

# add pitch to notes with -1 pitch
# analysis by "median note over bar", "key" and "bar chord"
def bar_add_pitch(bar_info, key_predict, chord_predict):
    avg_pitch = np.median([note for note in bar_info[:, 0] if note != -1])
    notes_score = keys_notes[key_predict] + chords_notes[chord_predict]
#     print("notes score:\t%s" % notes_score)

    SEARCH_PITCH_RANGE = 12
    min_pitch = (int)(np.max((0, np.min((np.floor(avg_pitch - SEARCH_PITCH_RANGE), np.min([note for note in bar_info[:, 0] if note != -1]))))))
    max_pitch = (int)(np.min((127, np.max((np.floor(avg_pitch + SEARCH_PITCH_RANGE), np.max(bar_info[:, 0]))))))

    notes_priority = np.zeros(max_pitch - min_pitch + 1)
    for i in range(0, max_pitch - min_pitch + 1):
        notes_priority[i] = 1 / (np.abs(avg_pitch - (i + min_pitch)) + SEARCH_PITCH_RANGE) * notes_score[(i + min_pitch) % 12]

#     print(notes_priority)
        
    avail_notes = np.argsort(-notes_priority) + min_pitch
#     print(avail_notes)

    for empty_idx in np.argwhere(bar_info[:, 0] == -1).flatten():
        notes_on = []
        for note_info in [note_info for note_info in bar_info if note_info[0] != -1]:
            bool_overlap = not(note_info[2] >= (bar_info[empty_idx, 2] + bar_info[empty_idx, 3]) or (note_info[2] + note_info[3]) <= bar_info[empty_idx, 2])
            if (note_info[0] not in notes_on and bool_overlap):
                notes_on.append(note_info[0])
        i = 0
#         print(notes_on)
        while (avail_notes[i] in notes_on):
            i += 1
            if i == avail_notes.shape[0]:
                i = 0
                break
        bar_info[empty_idx, 0] = avail_notes[i]
        
    return bar_info
    
def add_pitch(midi_info, mood, key_predict, chords_predict):
    if mood == MOOD_ANGRY or mood == MOOD_HAPPY:
        for bar_idx in range(midi_info.shape[0]):
            for channel_idx in range(MAX_NUM_OF_CHANNELS):
                if (-1 in midi_info[bar_idx, channel_idx][:, 0]):
                    midi_info[bar_idx, channel_idx] = bar_add_pitch(midi_info[bar_idx, channel_idx], key_predict, chords_predict[bar_idx])


# get the overall statistic of dynamics in each channel ([min, max, median])
def get_dynamics_info(midi_info):
    dynamics_info = np.zeros((MAX_NUM_OF_CHANNELS, 4), dtype=np.int16)
    for channel_idx in range(MAX_NUM_OF_CHANNELS):
        notes_dynamics = []
        for bar_idx in range(midi_info.shape[0]):
            for bar_info in midi_info[bar_idx, channel_idx]:
                notes_dynamics.append(bar_info[1])
        if len(notes_dynamics) > 0:
            dynamics_info[channel_idx] = [np.min(notes_dynamics), np.max(notes_dynamics), np.median(notes_dynamics), np.std(notes_dynamics)]
    return dynamics_info

def get_dynamics_lookup_table(dynamics_info, wide_factor, avg_addition_boost, avg_multiply_boost):
    dynamics_lookup_table = np.zeros((MAX_NUM_OF_CHANNELS, 128), dtype=np.int16)
    for channel_idx in range(MAX_NUM_OF_CHANNELS):
        dmin = dynamics_info[channel_idx, 0]
        dmax = dynamics_info[channel_idx, 1]
        dmed = dynamics_info[channel_idx, 2]
        for i in range(dmin, dmax + 1):
            note_velocity = 0
            if i < dmed:
                note_velocity = (int)(dmed - (dmed - dmin) * np.power((dmed - i) / (dmed - dmin), wide_factor[channel_idx]) * avg_multiply_boost[channel_idx])
            else:
                if dmax == dmed:
                    note_velocity = i
                else:
                    note_velocity = (int)(dmed - (dmed - dmax) * np.power((dmed - i) / (dmed - dmax), wide_factor[channel_idx]) * avg_multiply_boost[channel_idx])
            dynamics_lookup_table[channel_idx, i] = np.max((0, np.min((127, note_velocity + avg_addition_boost[channel_idx]))))
    return dynamics_lookup_table

def get_dynamics_angry_factors(dynamics_info, wide_para_1=0.3, wide_para_2=0.1, addition_para_1=10, multiply_para_1=0.2, multiply_para_2=0):
    dynaics_angry_factors = np.zeros((MAX_NUM_OF_CHANNELS, 3))
    dynaics_angry_factors[:, 0] = 1 - 1 / (dynamics_info[:, 3] * wide_para_1 + wide_para_2 + 1)
    dynaics_angry_factors[:, 1] = np.repeat(addition_para_1, MAX_NUM_OF_CHANNELS)
    dynaics_angry_factors[:, 2] = 1 + multiply_para_1 / (dynamics_info[:, 3] + multiply_para_2 + 1)
    return dynaics_angry_factors

def get_dynamics_calm_factors(dynamics_info, wide_para_1=0.8, addition_para_1=-10, multiply_para_1=0.2, multiply_para_2=0.5):
    dynaics_calm_factors = np.zeros((MAX_NUM_OF_CHANNELS, 3))
    dynaics_calm_factors[:, 0] = 1 + dynamics_info[:, 3] * wide_para_1
    dynaics_calm_factors[:, 1] = np.repeat(addition_para_1, MAX_NUM_OF_CHANNELS)
    dynaics_calm_factors[:, 2] = 1 / (dynamics_info[:, 3] * multiply_para_1 + 1) * (1 - multiply_para_2) + multiply_para_2
    return dynaics_calm_factors
    
def get_dynamics_happy_factors(dynamics_info, wide_para_1=5, wide_para_2=0.8, wide_para_3=0.5, addition_para_1=0, multiply_para_1=0.2, multiply_para_2=0):
    dynamics_happy_factors = np.zeros((MAX_NUM_OF_CHANNELS, 3))
    for i in range(MAX_NUM_OF_CHANNELS):
        if dynamics_info[i, 3] < wide_para_1:
            dynamics_happy_factors[i, 0] = 1 - wide_para_2 * np.power((wide_para_1 - dynamics_info[i, 3]) / wide_para_1, wide_para_2)
        else:
            dynamics_happy_factors[i, 0] = (1 - 1 / ((dynamics_info[i, 3] - wide_para_1) * wide_para_3 + 1)) * wide_para_2 + 1
    dynamics_happy_factors[:, 0] = 1 - 1 / (dynamics_info[:, 3] * wide_para_1 + wide_para_2 + 1)
    dynamics_happy_factors[:, 1] = np.repeat(addition_para_1, MAX_NUM_OF_CHANNELS)
    dynamics_happy_factors[:, 2] = 2 / (dynamics_info[:, 3] + 5) + 0.8
    return dynamics_happy_factors

def dynamics_angry(bar_info, dynamic_info, angry_dynamic_lookup_table, bar_notes_priority):
    if len(bar_info) == 0:
        return np.array([])    
    if np.min(bar_info[:,1]) != np.max(bar_info[:,1]) or len(bar_info) == 1:
        return angry_dynamic_lookup_table[[(int)(velocity) for velocity in bar_info[:, 1]]]
    else:
        dynamics_new = np.array([
            np.round(dynamic_info[0] + (dynamic_info[1] - dynamic_info[0]) / 5),
            np.round(dynamic_info[0] + (dynamic_info[1] - dynamic_info[0]) / 5 * 2),
            np.round(dynamic_info[0] + (dynamic_info[1] - dynamic_info[0]) / 5 * 3),
            np.round(dynamic_info[0] + (dynamic_info[1] - dynamic_info[0]) / 5 * 4),
            np.round(dynamic_info[0] + (dynamic_info[1] - dynamic_info[0]) / 5 * 4)
        ], dtype=np.int8)
#         print(dynamics_new)
        return angry_dynamic_lookup_table[dynamics_new[bar_notes_priority]]

def dynamics_calm(bar_info, calm_dynamic_lookup_table):
    return calm_dynamic_lookup_table[[(int)(velocity) for velocity in bar_info[:, 1]]]

def dynamics_happy(bar_info, happy_dynamic_lookup_table):
    return happy_dynamic_lookup_table[[(int)(velocity) for velocity in bar_info[:, 1]]]

def change_dynamics(midi_info, mood):
    dynamics_info = get_dynamics_info(midi_info)

    if mood == 0:
        dynamics_factors = get_dynamics_angry_factors(dynamics_info)
        dynamics_lookup_table = get_dynamics_lookup_table(dynamics_info, wide_factor=dynamics_factors[:, 0], avg_addition_boost=dynamics_factors[:, 1], avg_multiply_boost=dynamics_factors[:, 2])
        for bar_idx in range(midi_info.shape[0]):
            for channel_idx in range(MAX_NUM_OF_CHANNELS):
                bar_notes_priority = get_bar_notes_priority(midi_info[bar_idx, channel_idx], key_predict, chords_predict[bar_idx])
                new_dynamics = dynamics_angry(midi_info[bar_idx, channel_idx], dynamics_info[channel_idx], dynamics_lookup_table[channel_idx], bar_notes_priority)
                if len(new_dynamics) != 0:
                    midi_info[bar_idx, channel_idx][:, 1] = new_dynamics
    elif mood == 1 or mood == 3:
        dynamics_factors = get_dynamics_calm_factors(dynamics_info)
        dynamics_lookup_table = get_dynamics_lookup_table(dynamics_info, wide_factor=dynamics_factors[:, 0], avg_addition_boost=dynamics_factors[:, 1], avg_multiply_boost=dynamics_factors[:, 2])
        for bar_idx in range(midi_info.shape[0]):
            for channel_idx in range(MAX_NUM_OF_CHANNELS):
                new_dynamics = dynamics_calm(midi_info[bar_idx, channel_idx], dynamics_lookup_table[channel_idx])
                if len(new_dynamics) != 0:
                    midi_info[bar_idx, channel_idx][:, 1] = new_dynamics
    elif mood == 2:
        dynamics_factors = get_dynamics_happy_factors(dynamics_info)
        dynamics_lookup_table = get_dynamics_lookup_table(dynamics_info, wide_factor=dynamics_factors[:, 0], avg_addition_boost=dynamics_factors[:, 1], avg_multiply_boost=dynamics_factors[:, 2])
        for bar_idx in range(midi_info.shape[0]):
            for channel_idx in range(MAX_NUM_OF_CHANNELS):
                new_dynamics = dynamics_happy(midi_info[bar_idx, channel_idx], dynamics_lookup_table[channel_idx])
                if len(new_dynamics) != 0:
                    midi_info[bar_idx, channel_idx][:, 1] = new_dynamics
        
def get_module_vector(prev_key, new_key):
    module_vector = np.zeros((128), dtype=np.int8)
    
    tonic_note = prev_key % 12
    prev_key_type = prev_key // 12
    new_key_type = new_key // 12
    
    if prev_key_type == 0:
        # Major to natural minor
        if new_key_type == 1:                    
            range_chain = itertools.chain(range(tonic_note - 8, 128, 12), range(tonic_note - 3, 128, 12), range(tonic_note - 1, 128, 12))
            for i in range_chain:
                if i >= 0:
                    module_vector[i] -= 1
        # Major to harmonic minor
        if new_key_type == 2:                    
            range_chain = itertools.chain(range(tonic_note - 8, 128, 12), range(tonic_note - 3, 128, 12))
            for i in range_chain:
                if i >= 0:
                    module_vector[i] -= 1
                    
    if prev_key_type == 1:
        # natural minor to Major
        if new_key_type == 0:
            range_chain = itertools.chain(range(tonic_note - 9, 128, 12), range(tonic_note - 4, 128, 12), range(tonic_note - 2, 128, 12))
            for i in range_chain:
                if i >= 0:
                    module_vector[i] += 1
        # natural minor to harmonic minor
        if new_key_type == 2:
            for i in range(tonic_note - 2, 128, 12):
                if i >= 0:
                    module_vector[i] += 1
        # harmonic minor to natural minor
        if new_key_type == 1:
            for i in range(tonic_note - 1, 128, 12):
                if i >= 0:
                    module_vector[i] -= 1
                    
    # change tonic key
    module_vector += ((new_key % 12 - prev_key % 12) + 6) % 12 - 6
                        
    return module_vector

CIRCLE_OF_FIFTH = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=np.int8)
MOOD_MODULE_TYPE = np.array([
    [1, 1, 2],
    [0, 1, 1],
    [0, 0, 0],
    [2, 2, 2]
])
MOOD_MODULE_TONIC = np.array([-1, 8, 4, 0], dtype=np.int8)

def get_key_to_module(prev_key, mood):
    MAX_MOOD_DIST = 3
    prev_key_tonic = prev_key % 12
    prev_key_type = prev_key // 12
    new_key_type = MOOD_MODULE_TYPE[mood, prev_key_type]
    
    if mood == MOOD_ANGRY:
        return new_key_type * 12 + prev_key_tonic
    else:
        minor_shift = 0
        if new_key_type != 0:
            minor_shift = 3
        circle_idx = np.where(CIRCLE_OF_FIFTH == (prev_key_tonic + minor_shift) % 12)[0][0]
        
        mood_direction = (MOOD_MODULE_TONIC[mood] - circle_idx + 12) % 12 <= (circle_idx - MOOD_MODULE_TONIC[mood] + 12) % 12
        mood_move = -1
        if mood_direction:
            mood_move = 1
                
        tonic_dists = np.array([[CIRCLE_OF_FIFTH[(circle_idx + minor_shift) % 12] - prev_key_tonic, circle_idx]])
        for i in range(MAX_MOOD_DIST):
            if circle_idx == MOOD_MODULE_TONIC[mood]:
                break
            circle_idx = (circle_idx + mood_move) % 12
            tonic_dists = np.vstack((tonic_dists, [CIRCLE_OF_FIFTH[(circle_idx + minor_shift) % 12] - prev_key_tonic, circle_idx]))
            
        for i in range(tonic_dists.shape[0]):
            if tonic_dists[i, 0] < -6:
                tonic_dists[i, 0] += 12
            elif tonic_dists[i, 0] >= 6:
                tonic_dists[i, 0] -= 12
        
        if mood == MOOD_CALM or mood == MOOD_SAD:
            circle_idx = tonic_dists[np.argmin(tonic_dists[:, 0]), 1]
        elif mood == MOOD_HAPPY:
            circle_idx = tonic_dists[np.argmax(tonic_dists[:, 0]), 1]
                    
        return new_key_type * 12 + CIRCLE_OF_FIFTH[(circle_idx + minor_shift) % 12]

def module_key(midi_info, module_vector):
    for channel_idx in range(MAX_NUM_OF_CHANNELS):
        if channel_idx == DRUM_CHANNEL:
            continue
        for bar_idx in range(midi_info.shape[0]):
            for info_idx in range(midi_info[bar_idx, channel_idx].shape[0]):
                midi_info[bar_idx, channel_idx][info_idx, 0] += module_vector[(int)(midi_info[bar_idx, channel_idx][info_idx, 0])]
    return midi_info

def info2midi(midi_info, msg_non_note, ticks_per_beat):

    UINT32_MAX = 4294967296 - 1

    channels_block = np.empty((MAX_NUM_OF_CHANNELS), dtype=np.object)
    for i in range(MAX_NUM_OF_CHANNELS):
        channels_block[i] = np.empty((0, 5), dtype=np.uint32)

    ticks_per_beat = midi_file.ticks_per_beat

    for channel_idx in range(MAX_NUM_OF_CHANNELS):
        for bar_idx in range(midi_info.shape[0]):
            bar_tick = ticks_per_beat * bars_beat[bar_idx]
            for info in midi_info[bar_idx, channel_idx]:
                if info[6] == 0:
                    channels_block[channel_idx] = np.vstack((channels_block[channel_idx], [(bar_idx + info[2]) * bar_tick, 0, channel_idx, info[0], info[1]]))
                    channels_block[channel_idx] = np.vstack((channels_block[channel_idx], [(bar_idx + info[2] + info[3]) * bar_tick, 1, channel_idx, info[0], info[1]]))
                else:
                    if info[2] == 0:
                        channels_block[channel_idx] = np.vstack((channels_block[channel_idx], [(bar_idx + info[2] + info[3]) * bar_tick, 1, channel_idx, info[0], info[1]]))
                    else:
                        channels_block[channel_idx] = np.vstack((channels_block[channel_idx], [(bar_idx + info[2]) * bar_tick, 0, channel_idx, info[0], info[1]]))
            channels_block[channel_idx] = channels_block[channel_idx][channels_block[channel_idx][:,0].argsort()]
                
    first_msg = np.zeros((17, 2), dtype=np.uint32)
    # initialize the comparison array
    for channel_idx in range(MAX_NUM_OF_CHANNELS):
        if channels_block[channel_idx].shape[0] != 0:
            first_msg[channel_idx, 1] = channels_block[channel_idx][0, 0]
        else:
            first_msg[channel_idx, 1] = UINT32_MAX
    if msg_non_note.shape[0] != 0:
        first_msg[16, 1] = msg_non_note[0, 0]
    else:
        first_msg[16, 1] = UINT32_MAX


    midi_file_new = mido.MidiFile(type=0)
    midi_file_new.ticks_per_beat = midi_file.ticks_per_beat
    track_new = mido.MidiTrack()
    midi_file_new.tracks.append(track_new)

    total_ticks = 0

    while True:
        channel_to_append = np.argmin(first_msg[:, 1])
        if first_msg[channel_to_append, 1] == UINT32_MAX:
            break
        if channel_to_append < 16:
            info = channels_block[channel_to_append][first_msg[channel_to_append, 0]]

            tick = (int)(info[0])
            msg = mido.Message('note_on', channel=(int)(info[2]), note=(int)(info[3]), velocity=(int)(info[4]), time=0)
            if info[1] == 1:
                msg = mido.Message('note_off', channel=(int)(info[2]), note=(int)(info[3]), velocity=(int)(info[4]), time=0)

            if tick != total_ticks:
                msg.time = tick - total_ticks
                total_ticks = tick

            track_new.append(msg)

            first_msg[channel_to_append, 0] += 1
            if first_msg[channel_to_append, 0] < channels_block[channel_to_append].shape[0]:
                first_msg[channel_to_append, 1] = channels_block[channel_to_append][first_msg[channel_to_append, 0], 0]
            else:
                first_msg[channel_to_append, 1] = UINT32_MAX
        else:
            info = msg_non_note[first_msg[channel_to_append, 0]]

            tick = (int)(info[0])        
            msg = info[1]
            msg.time = 0

            if tick != total_ticks:
                msg.time = tick - total_ticks
                total_ticks = tick

            track_new.append(msg)

            first_msg[channel_to_append, 0] += 1
            if first_msg[channel_to_append, 0] < msg_non_note.shape[0]:
                first_msg[channel_to_append, 1] = msg_non_note[first_msg[channel_to_append, 0], 0]
            else:
                first_msg[channel_to_append, 1] = UINT32_MAX
    return midi_file_new
            
# input the file name
midi_string = 'sample2'
midi_file = mido.MidiFile('../sample_audio/%s.mid' % midi_string)

mood_input = 1

# get the beat and tempo in each bar
_, bars_tick, bars_beat, bars_tempo = get_bars_info(midi_file)

# separate notes to their relative bars
bars_note_info = get_bars_note_info(midi_file)

# generate the rhythm pattern[channel index, bar index]
# rhythm_pattern = get_rhythm_pattern(bars_note_info)

midi_info = get_midi_info(bars_note_info)

chords_predict = get_chords(midi_info)

key_predict, key_acc, chords_stat = get_key(chords_predict)

msg_non_note = get_msg_non_note(midi_file)
    
mood_change(mood_input, midi_info, chords_predict)
    
add_pitch(midi_info, mood_input, key_predict, chords_predict)

# with np.printoptions(precision=3, suppress=True):
#     print(midi_info[18:22, 5])
    
change_dynamics(midi_info, mood_input)

new_key = get_key_to_module(key_predict, mood_input)

module_vector = get_module_vector(key_predict, new_key)
module_key(midi_info, module_vector)

new_chords_predict = get_chords(midi_info)

new_key_predict, new_key_acc, new_chords_stat = get_key(new_chords_predict)

midi_file_new = info2midi(midi_info, msg_non_note, midi_file.ticks_per_beat)

midi_file_new.save('../sample_audio_export/%s_%s.mid' % (midi_string, MOODS_NAME[mood_input]))