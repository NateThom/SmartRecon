import sys
import os
import time

def main(argv):
	hashes = []
	inbase = '/home/jay/Desktop/5-minute/'
	outbase = '/home/jay/Desktop/5-minute-cleaned/'

	for folder in os.listdir(inbase):
		print(folder)
		for file in os.listdir(inbase + folder):
			inputfile = inbase + folder + '/' + file
			outputfile = outbase + folder + '/' + file

			command1 = 'tcprewrite --dlt=enet --infile=' + inputfile + ' ' + '--outfile=temp.pcap'
			time.sleep(.03)
			command2 = 'tcprewrite --enet-dmac=11:11:11:11:11:11 --enet-smac=11:11:11:11:11:11 --pnat=192.168.0.0/16:1.1.1.1/32 --infile=temp.pcap --outfile=' + outputfile
			time.sleep(.03)

			os.system(command1)
			os.system(command2)


if __name__ == '__main__':
    main(sys.argv)