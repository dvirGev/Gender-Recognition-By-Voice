import wave

# Open the source wave file in read mode
with wave.open('test-samples/27-124992-0002.wav', 'rb') as source:
    # Read the wave file's metadata
    num_channels = source.getnchannels()
    sample_width = source.getsampwidth()
    frame_rate = source.getframerate()
    num_frames = source.getnframes()

    # Read the wave file's raw audio data
    audio_data = source.readframes(num_frames)
    from struct import pack
    audio_data = pack('<' + ('h'*len(audio_data)), *audio_data)

# Open the destination wave file in write mode
with wave.open('destination_file.wav', 'wb') as dest:
    # Set the wave file's metadata
    dest.setnchannels(1)
    dest.setsampwidth(sample_width)
    dest.setframerate(frame_rate)
    dest.setnframes(num_frames)

    # Write the audio data to the wave file
    dest.writeframes(audio_data)
