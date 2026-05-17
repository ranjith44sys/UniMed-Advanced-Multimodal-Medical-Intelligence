import os
# Set cache dir BEFORE importing kagglehub
os.environ['KAGGLEHUB_CACHE'] = 'd:/KaggleCache'

import kagglehub

print("Starting download to:", os.environ['KAGGLEHUB_CACHE'])
path = kagglehub.dataset_download('paultimothymooney/breast-histopathology-images')
print('Download completed! Path:', path)
