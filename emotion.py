import os
import sys
import platform
import struct
import numpy as np
import pyaudio
import time

# Adjust this path to match your actual directory structure
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
api_dir = os.path.join(base_dir, 'emotion/OpenVokaturi-4-0/OpenVokaturi-4-0', 'api')
lib_dir = os.path.join(base_dir, 'emotion/OpenVokaturi-4-0/OpenVokaturi-4-0', 'lib', 'open', 'win')

sys.path.append(api_dir)

try:
    import Vokaturi
except ImportError:
    print(f"Error: Could not import Vokaturi. Please ensure the 'api' folder containing Vokaturi.py is in {api_dir}")
    sys.exit(1)

print("Loading library...")

if platform.system() == "Windows":
    if struct.calcsize("P") == 4:
        lib_path = os.path.join(lib_dir, "OpenVokaturi-4-0-win32.dll")
    else:
        lib_path = os.path.join(lib_dir, "OpenVokaturi-4-0-win64.dll")
    
    if not os.path.exists(lib_path):
        print(f"Error: Could not find the Vokaturi library at {lib_path}")
        sys.exit(1)
    
    Vokaturi.load(lib_path)
else:
    print("This script is configured for Windows. Please use the appropriate version for your OS.")
    sys.exit(1)

print("Analyzed by: %s" % Vokaturi.versionAndLicense())

p = pyaudio.PyAudio()
c_buffer = Vokaturi.float32array(10000)  # usually 1024 would suffice

def callback(in_data, frame_count, time_info, flag):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    c_buffer[0 : frame_count] = audio_data
    voice.fill_float32array(frame_count, c_buffer)
    return (in_data, pyaudio.paContinue)

print("Creating VokaturiVoice...")
sample_rate = 44100
buffer_duration = 10  # seconds
buffer_length = sample_rate * buffer_duration
voice = Vokaturi.Voice(
    sample_rate,
    buffer_length,
    1  # because fill() and extract() are in different threads
)

print("PLEASE START TO SPEAK...")
stream = p.open(
    rate=sample_rate,
    channels=1,
    format=pyaudio.paFloat32,
    input=True,
    output=False,
    start=True,
    stream_callback=callback
)

approximate_time_elapsed = 0.0  # will not include extraction time
time_step = 0.5  # seconds

try:
    while stream.is_active():  # i.e. forever
        time.sleep(time_step)

        quality = Vokaturi.Quality()
        emotionProbabilities = Vokaturi.EmotionProbabilities()
        voice.extract(quality, emotionProbabilities)
        approximate_time_elapsed += time_step
        
        if quality.valid:
            print(
                f"{approximate_time_elapsed:5.1f} time",
                f"{emotionProbabilities.neutrality*100:5.0f} N",
                f"{emotionProbabilities.happiness*100:5.0f} H",
                f"{emotionProbabilities.sadness*100:5.0f} S",
                f"{emotionProbabilities.anger*100:5.0f} A",
                f"{emotionProbabilities.fear*100:5.0f} F"
            )
except KeyboardInterrupt:
    print("\nStopping the stream...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stream closed and PyAudio terminated.")