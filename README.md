# Master os Science (later actual topic)
## Paper
Contains paper for my master thesis and all necessary files and images.

## Networks
Contains all scripts necessary for training networks and all trained models.

## Scripts
Contains all scripts used to process data from data sets.

1. extractFrames.py - extracts frames from videos in fixed time interval. Also creates adequate directories.
2. splitDatasets - takes two folders with faces and splits them to tests and training directories.
3. compressData.py - takes test and training folders and compresses them into npz file for cycleGan training.
4. prepareNPZ.py - takes test and training folders and compresses them into npz file for VAE training.
5. check_images_sizes.py - checks if all images has the same size.
6. rescale_images.py - changes images sizes to fixed size.