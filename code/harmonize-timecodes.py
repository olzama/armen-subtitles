import pysrt
from datetime import datetime, timedelta
import sys


'''
Given a path to an SRT file, ensure that the end of each subtitle segment
is the same as the start time of the next segment. 
'''
def harmonize_timecodes(input_path, output_path):
    subs = pysrt.open(input_path)
    for i, entry in enumerate(subs[:-1]):
        subs[i + 1].start = entry.end
    subs.save(output_path, encoding='utf-8')

if __name__ == '__main__':
    input_srt_path = sys.argv[1]
    output_srt_path = sys.argv[2]
    harmonize_timecodes(input_srt_path, output_srt_path)


