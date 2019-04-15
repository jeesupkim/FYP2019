import mido
import numpy as np

# return true if the message is a note
def is_note(msg):
    return msg.type == 'note_on' or msg.type == 'note_off'

# return true if the message indicates the start of a note
# return false if the message indicates the end of a note
def is_note_on(msg):
    return msg.type == 'note_on' and msg.velocity != 0

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
    MAX_NUM_OF_CHANNELS = 16
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
    

# input the file name
midi_string = 'sample1'
midi_file = mido.MidiFile('../sample_audio/%s.mid' % midi_string)

# get the beat and tempo in each bar
_, bars_tick, bars_beat, bars_tempo = get_bars_info(midi_file)

# separate notes to their relative bars
bars_note_info = get_bars_note_info(midi_file)

# generate the rhythm pattern[channel index, bar index]
# rhythm_pattern = get_rhythm_pattern(bars_note_info)

midi_info = get_midi_info(bars_note_info)







with np.printoptions(precision=3, suppress=True):
#     print(bars_tick)
#     print([bar_note_info[3] for bar_note_info in bars_note_info])
    print(midi_info[:,0][0:10])
#     print(rhythm_pattern[0, 0])
#     print(bars_note_info[0])