# Research on methods of changing objects in images using Deepfake technology

Deepfake is a technique from image-to-image translation class of problems, designed to combine and overlay objects in images or videos creating deceptively realistic counterfeits. This paper analyzes and compares four leading methods used for deepfake generation: autoencoders, variational autoencoders, variational autoencoders generative adversarial networks and cycle generative adversarial networks, in problem of face-to-face conversion. Due to the lack of numerical methods for deepfake comparison, all obtained results were assessed in a visual evaluation process. Variational autoencoders technique has proved to be the most efficient one in facial-deepfake generation problem. The worst results were obtained from CycleGAN method, which proved to be unfitted for geometric changes and shape transformations. It was concluded that VAE-GAN technique is the one with the greatest potential, as in case of feature maps with better quality, this approach could provide deceivingly resembling deepfakes.

## Paper

Contains paper for my master thesis and all necessary files and images.

## Colab Notebooks

Contains all scripts necessary for training networks for each method.

## Scripts

Contains all scripts used to process data from data sets.

1. extractFrames.py - extracts frames from videos in fixed time interval. Also creates adequate directories.
2. splitDatasets - takes two folders with faces and splits them to tests and training directories.
3. compressData.py - takes test and training folders and compresses them into npz file for cycleGan training.
4. prepareNPZ.py - takes test and training folders and compresses them into npz file for VAE training.
5. check_images_sizes.py - checks if all images has the same size.
6. rescale_images.py - changes images sizes to fixed size.
7. generate_from_models - generates deepfakes form given, pre-trained models for AE, VAE and VAE-GAN methods.
8. generate_from_cyclegan - generates deepfakes form given, pre-trained models for CycleGAN method.

## Test_Images

Contains pictures used to generate final results for paper.

## Models

Contains models of trained networks for AE and VAE methods.
