import json
import os
import math
import librosa

DATASET_PATH = "/content/MyDrive/My Drive/genres"
JSON_PATH = "data.json" #storing current working folder ,here we want to store all the mfccs

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

#segments to chop the tracks
#not saving as track.saving as segments
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """
   
    # dictionary to store mapping, labels, and MFCCs
    # mapping : mapping diffrent genres to levels like ['classical','blues']
    # mfccs are training inputs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # we need a constant num of mfccs per segment
    #we calculate mfccs per hop length
    # ceil makes 1.2 ->2 
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    # dirpath is the current folder genres,dirnames are subfolders in genre,filenames are audio files
    # count is needed as we need lebels and labels will be 0,1,2...

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure we're processing a genre sub-folder level
        # we are not in the dataset folder level path.we need to be inside the genere folder
        if dirpath is not dataset_path:
            
            # save genre label (i.e., sub-folder name) in the mapping
            # dirpath.split("/")[-1] : dirpath is like genre/blues convreted to ['genre','blues']
            # -1 means will take the last index which is blues..for mapping we need this name
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE) #loading wav file

                # process all segments of audio file,extracting mfccs,storing data
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    # we will slide right
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    # mfcc(signal[start:finish], : mfcc is taken from slice of signal
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    # data["labels"].append(i-1) as we aree in wav file,if we want the level we need to be in prev
                    # if i is wav file,i-1 is blues/classical and this the level
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
         
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH,num_segments=10)
