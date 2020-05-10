import os
import sys
import pandas as pd

model = sys.argv[1]
log_file = 'logs/%s-annotate.log'%model
annotations_df = pd.read_csv('../data/action_annotation.csv')

DEBUG = False

class interval(object):
	
	def __init__(self, start, end):
		self.start = start
		self.end = end

	@property
	def length(self):
		return self.end - self.start

def get_hits(p_interval, time_intervals):

	## Base case I
	if not p_interval.length:
		return 0, 0

	## Base case II
	if not len(time_intervals):
		return 0, p_interval.length

	## Ending is lesser than the start of time_intervals[0]: Complete false hit
	if p_interval.end < time_intervals[0].start:
		return 0, p_interval.length

	## Starting is more than the end of time_intervals[0]: Complete false hit, pop time_intervals[0] and recursively get_hits
	if p_interval.start > time_intervals[0].end:
		_f = time_intervals[0].length
		del time_intervals[0]
		t, f = get_hits(p_interval, time_intervals)
		return t, _f + f

	## Partial overlap with start of time_intervals: Update true hit and false hit
	if p_interval.start < time_intervals[0].start:

		## time_intervals[0] ending lies beyond p_interval end: Good case! Wrap up with the true hits
		if p_interval.end < time_intervals[0].end:
			t, f = (p_interval.end - time_intervals[0].start), (time_intervals[0].start - p_interval.start)
			time_intervals[0].start = p_interval.end
			return t, f

		## Else, update the true hit count, remove the head of time_intervals, and recursively calculate hits
		_true_hit, _false_hit = time_intervals[0].length, (time_intervals[0].start - p_interval.start)
		p_interval.start = time_intervals[0].end
		del time_intervals[0]
		t, f = get_hits(p_interval, time_intervals)
		return _true_hit + t, _false_hit + f

	## p_interval start is more than or equal to time_intervals[0] start

	## p_interval end is less than time_intervals[0] end: Good case! Wrap up with the true hits
	if p_interval.end < time_intervals[0].end:
		time_intervals[0].start = p_interval.end
		return p_interval.length, 0

	## Else, update the true hit count, remove the head of time_intervals, and recursively calculate hits
	_true_hit, _false_hit = time_intervals[0].end - p_interval.start, 0
	p_interval.start = time_intervals[0].end
	del time_intervals[0]
	t, f = get_hits(p_interval, time_intervals)
	return _true_hit + t, _false_hit + f

with open(log_file, 'r') as l:

	total_true_hits, total_false_hits, total_gt_hits, total_frame_count = 0, 0, 0, 0
	video_name = l.readline().split('.')[0]
	while True:
		
		## Read in 'investigating' (start, end) tuples, sorted disjoint intervals
		investigate_df = annotations_df[(annotations_df['Scoring'] == video_name) & (annotations_df['Behaviour'] == 'Investigating')]
		time_intervals = [interval(int(r['Start_Frame']), int(r['Stop_Frame'])) for _, r in investigate_df.iterrows()]
		## Iterate through the log and update true hits and false hits count at frame level
		true_hits, false_hits, gt_hits, frame_count = 0, 0, sum([t.length for t in time_intervals]), int(annotations_df[(annotations_df['Scoring'] == video_name) & (annotations_df['Value'] == 'End')]['Start_Frame'])
		data = l.readline().split()
		while len(data) == 3:
			frame_id, prediction = int(data[0]), data[1]
			p_interval = interval(frame_id-60, frame_id)

			if prediction == 'investigate':
				t, f = get_hits(p_interval, time_intervals)
				if DEBUG:
					print("=> Interval [%d-%d]: (%d, %d)"%(p_interval.start, p_interval.end, t, f))
				true_hits += t; false_hits += f 

			data = l.readline().split()
		# print("%s: %f (%d/%d) %f (%d/%d)"%(video_name, true_hits/gt_hits, true_hits, gt_hits, false_hits/(frame_count - gt_hits), false_hits, (frame_count - gt_hits)))
		print("%s %f %f"%(video_name.split('_')[2], true_hits/gt_hits, false_hits/(frame_count - gt_hits)))
		total_true_hits += true_hits
		total_false_hits += false_hits
		total_gt_hits += gt_hits
		total_frame_count += frame_count

		if len(data) > 1: 
			break

		video_name = data[0].split('.')[0]

	print("Final: %f (%d/%d) %f (%d/%d)"%(total_true_hits/total_gt_hits, total_true_hits, total_gt_hits, total_false_hits/(total_frame_count - total_gt_hits), total_false_hits, (total_frame_count - total_gt_hits)))
