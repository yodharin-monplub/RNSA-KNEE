"""Exam-level label counts derivable from the fastMRI+ knee annotations (data/external/fastmri_plus)."""
import pandas as pd

boxes = pd.read_csv('data/external/fastmri_plus/knee.csv')
files = pd.read_csv('data/external/fastmri_plus/knee_file_list.csv', header=None)[0]
print('annotated exams', files.nunique(), '| exams with any box', boxes.file.nunique())
counts = boxes.groupby('label').file.nunique().sort_values(ascending=False)
print(counts.to_string())
