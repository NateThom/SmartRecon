
import os
from itertools import combinations
import itertools
import numpy as np
import os
import os.path
import random
import subprocess
import sys
import time





def splitter(input_pcap, outputbase, increment):

	command = "tshark -r " + input_pcap + " -T fields -e frame.time_relative"
	proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()
	# print(out)
	
	splitted = out.split(b"\n")
	last_second = int(float(splitted[-2]))
	# window_insecond = 60*10
	# window_insecond = 60 * increment
	window_insecond = 60 * 1

	counter = 1
	if last_second > window_insecond:
 
		# for i in range(0, last_second, 60*10):		
		# for i in range(0, last_second, 60 * increment):
		for i in range(0, last_second, 60 * 1):

			start_time = i 
			end_time = i + window_insecond
			output_pcap=outputbase+input_pcap.split("/")[-1].split(".pcap")[0]+"-"+str(counter)+".pcap"

			command = "tshark -r " + input_pcap
			command = command + " -Y 'frame.time_relative >= "
			command = command + str(start_time)
			command = command + " and frame.time_relative <= "
			command = command + str(end_time) + "' -w "
			command = command + output_pcap

			os.system(command)

			# if((end_time + 60*10) > last_second):
			# if((end_time + 60 * increment) > last_second):
			if((end_time + 60 * 1) > last_second):
				break
			counter = counter + 1
			


	print ("finished",input_pcap)





# base = '/home/jay/Desktop/10-minute/'
# for folder in os.listdir(base):
# 	for file in os.listdir(base + '/' + folder):

# 		outputbase = "/home/jay/Desktop/increments_9/" + folder + '/' + file.split("_")[0]
# 		print(outputbase)

# 		# path_to_file = base + '/' + folder + '/' + file

# 		# print(path_to_file)
# 		# splitter(path_to_file, outputbase)

inpath = '/home/jay/Desktop/24-hour/'
# outpath = '/home/jay/Desktop/10-minute/'
outpath = '/home/jay/Desktop/'
counter = 10

# while counter > 0:
for folder in os.listdir(inpath):
	for file in os.listdir(inpath + folder):
		input_pcap = inpath + folder + '/' + file
		# outputbase = outpath + str(counter) + '-minute/' + folder + '/'
		outputbase = outpath + '1-minute/' + folder + '/'
		splitter(input_pcap, outputbase, counter)
		# print(input_pcap, outputbase)
	# counter -= 1


	
