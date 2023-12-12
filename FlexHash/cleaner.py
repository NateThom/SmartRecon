import sys
import os
import time

def main(argv):
	hashes = []
	inbase = 'plug-8/per_packet_no_activity/'
	outbase = 'plug-8/per_packet_no_activity_cleaned/'

	for file in os.listdir(inbase):
		inputfile = inbase + file
		outputfile = outbase + file

		print(inputfile + ' : ' + outputfile)

		command1 = 'tcprewrite --dlt=enet --infile=' + inputfile + ' ' + '--outfile=temp.pcapng'
		os.system(command1)
		# time.sleep(.03)

		command2 = 'tcprewrite --enet-dmac=11:11:11:11:11:11 --enet-smac=11:11:11:11:11:11 --pnat=192.168.0.0/16:1.1.1.1/32 --infile=temp.pcapng --outfile=' + outputfile
		os.system(command2)
		# time.sleep(.03)


if __name__ == '__main__':
    main(sys.argv)
