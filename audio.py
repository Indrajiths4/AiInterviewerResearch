# import numpy as np
# import matplotlib.pyplot as plt
# from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import ShortTermFeatures

# def analyze_audio(file_path):
#     # Read audio file
#     Fs, x = audioBasicIO.read_audio_file(file_path)

#     # Extract short-term features
#     F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

#     # Calculate average features over the entire audio for analysis
#     avg_features = np.mean(F, axis=1)

#     # Print relevant features
#     print("Average Features:")
#     for i in range(len(avg_features)):
#         print(f"{f_names[i]}: {avg_features[i]}")

#     # Plotting the first feature (energy) over time for visualization
#     plt.figure(figsize=(10, 6))
#     plt.plot(F[0, :], label='Energy')
#     plt.title('Energy over Time')
#     plt.xlabel('Frame Number')
#     plt.ylabel('Energy')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Example usage
# analyze_audio("audio.wav")














import numpy as np
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

def analyze_audio_advanced(file_path):
    # Read audio file
    Fs, x = audioBasicIO.read_audio_file(file_path)

    # Extract short-term features
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

    # Calculate average features
    avg_features = np.mean(F, axis=1)

    # Print all feature names
    print("Available features:")
    for i, name in enumerate(f_names):
        print(f"{i}: {name}")

    # Function to find features containing a keyword
    def find_features(keyword):
        return [name for name in f_names if keyword.lower() in name.lower()]

    # Print relevant features
    print("\nTone-related features:")
    for feature in find_features('spectral') + find_features('mfcc'):
        idx = f_names.index(feature)
        print(f"{feature}: {avg_features[idx]}")

    print("\nPitch-related features:")
    for feature in find_features('zero') + find_features('chroma'):
        idx = f_names.index(feature)
        print(f"{feature}: {avg_features[idx]}")

    print("\nRhythm-related features:")
    for feature in find_features('energy') + find_features('entropy'):
        idx = f_names.index(feature)
        print(f"{feature}: {avg_features[idx]}")

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Spectral Centroid (related to tone)
    spectral_features = find_features('spectral')
    if spectral_features:
        axs[0].plot(F[f_names.index(spectral_features[0]), :])
        axs[0].set_title(f'{spectral_features[0]} over Time')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel(spectral_features[0])

    # Zero Crossing Rate (related to pitch)
    zcr_features = find_features('zero')
    if zcr_features:
        axs[1].plot(F[f_names.index(zcr_features[0]), :])
        axs[1].set_title(f'{zcr_features[0]} over Time')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel(zcr_features[0])

    # Energy (can be related to rhythm)
    energy_features = find_features('energy')
    if energy_features:
        axs[2].plot(F[f_names.index(energy_features[0]), :])
        axs[2].set_title(f'{energy_features[0]} over Time')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel(energy_features[0])

    plt.tight_layout()
    plt.show()

# Example usage
analyze_audio_advanced("audio.wav")