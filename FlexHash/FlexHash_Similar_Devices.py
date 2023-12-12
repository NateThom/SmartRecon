# import modules
import sys
import os
import itertools
import statistics
from nltk import ngrams
import hashlib
from itertools import combinations
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm


# class: Simhash
class Simhash:
    # call constructor
    def __init__(self, acc_size, window_size, comb_size):

        # set internal variables
        self.acc_size = acc_size
        self.window_size = window_size

        # set combination size, if bigger than window, take window size instead
        if window_size > comb_size:
            self.comb_size = comb_size
        else:
            self.comb_size = window_size
        self.accumulator = [0] * self.acc_size
        self.window = [-1] * window_size
        self.hash_const = []

        np.random.seed(30)
        hash_const = np.random.choice(range(acc_size), size=acc_size, replace=False)
        for item in hash_const:
            self.hash_const.append(item)

    # class method: hash combinations
    def hashenator(self, s):
        result = 1
        for item in s:
            result = item * result
        return (result + 3) % self.acc_size

    # class method: calc_digest
    def calc_digest(self, str1, label):

        # set a local counter to iterate
        counter = 0

        # iterate through string with sliding window of x bytes
        for x in range(counter, len(str1)):
            self.window = str1[x:x + self.window_size]
            counter += 1

            # take each combination of size n
            for combination in itertools.combinations(self.window, self.comb_size):
                # call hash function
                val = self.hashenator(combination)

                # get index of value in hash_const array
                ind = self.hash_const.index(val)

                # increment value at accumulator
                self.accumulator[ind] += 1

        # find median value in accumulator
        median_value = statistics.median(self.accumulator)

        # assign 1 or 0 based on median
        for x in range(0, self.acc_size):
            if self.accumulator[x] <= median_value:
                self.accumulator[x] = 0
            else:
                self.accumulator[x] = 1

        # create list to hold final feature vector
        hash_list = []

        # iterate through each byte and convert to decimal
        for x in range(0, self.acc_size, 8):
            new_string = ''
            this_byte = self.accumulator[x:x + 8]

            for bit in this_byte:
                new_string += str(bit)
            # hash_list.append(int(new_string,2))
            entry = int(new_string, 2)
            hash_list.append(str(entry))

        # append label
        hash_list.append(label)

        # return final feature vector
        return hash_list


def wrapper_function(filename, label, acc_size, window_size, ngram_size):
    sim = Simhash(acc_size, window_size, ngram_size)

    # open input file and read to str1 as bytes object
    with open(filename, 'rb') as pcap:
        # read in file...
        str1 = pcap.read()

        # break into list of bytes objects
        strlist = [x for x in str1]

        hashed = sim.calc_digest(strlist, label)

        # call sim.calc_digest
        return hashed


# main function
def main(argv):

    # accum_params = [128, 256, 512, 1024]
    accum_params = [1024]

    # window_params = [4, 5, 6]
    window_params = [6]

    path_to_output = "/storage/nate/SmartRecon/FlexHash/similar_devices_Hashes/"
    if not os.path.isdir(path_to_output):
        os.mkdir(path_to_output)

    path_to_outer_directory = "/storage/nate/SmartRecon/FlexHash/similar_devices/"

    # plug_dir_list = [f"plug-{i}" for i in range(1, 9)]
    #light_dir_list = [f"light-{i}" for i in range(1, 9)]
    #cam_dir_list = [f"cam-{i}" for i in range(1, 9)]
    # cam_allwinner_dir_list = [f"cam_allwinner_streaming-{i}" for i in range(1, 9)]
    plug_updated_dir_list = [f"plug_updated-{i}" for i in range(1, 9)]
    #device_list = plug_dir_list + light_dir_list + cam_dir_list
    #device_list = plug_dir_list + light_dir_list
    # device_list = plug_dir_list
    device_list = plug_updated_dir_list

    # packet_dir_list = ["per_packet_no_activity", "per_packet_no_activity_cleaned"]
    # packet_dir_list = ["per_packet_no_activity"]
    
    # cleaned_list = ["_cleaned", ""]
    c_uc_list = ["_cleaned"]

    for accum in tqdm(accum_params):
        for window in window_params:
            combo_params = list(range(2, window+1))
            for combo in combo_params:
                for device in device_list:
                    for c_uc in c_uc_list:
                        
                        file_list = os.listdir(path_to_outer_directory + device + "/" + "per_packet_no_activity" + c_uc)
                        file_list = [f"{path_to_outer_directory}{device}/{'per_packet_no_activity' + c_uc}/{i}" for i in file_list]
                        # file_list = file_list[:1]
                        dataset_params = zip(
                            file_list,
                            itertools.repeat(device, len(file_list)),
                            itertools.repeat(accum, len(file_list)),
                            itertools.repeat(window, len(file_list)),
                            itertools.repeat(combo, len(file_list))
                        )

                        with multiprocessing.Pool() as p:
                            output = p.starmap(wrapper_function, dataset_params)

                        #try:
                        output_df = pd.DataFrame(output)
                        #num_cols = len(output_df.axes[0])
                        #column_names = list(range(len(output[0]) - 1))
                        #column_names.append("label")
                        #output_df.columns = column_names
                        
                        output_df.to_csv(f"{path_to_output}{device.split('-')[0]}{c_uc}/{device}_{accum}_{window}_{combo}.csv", index=False)
                        # print(f"Output: {path_to_output}{device.split('-')[0]}{c_uc}/{device}_{accum}_{window}_{combo}.csv")
                        # output_df.to_csv(f"{path_to_output}{device}_{packet_dir}_{accum}_{window}_{combo}.csv", index=False)
                        
                        #except:
                            #print("***FAILED***")
                            #print(f"{path_to_output}{device}_{packet_dir}_{accum}_{window}_{combo}")
                            #print()
                            #print(f"{device}_{packet_dir}")

if __name__ == '__main__':
    main(sys.argv)
