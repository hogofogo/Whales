# whales model

## Overview

https://www.kaggle.com/c/whale-categorization-playground

From Kaggle: This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an Id. The challenge is to predict the whale Id of images in the test set. What makes this such a challenge is that  there are only a few examples for each of 3,000+ whale Ids.
=======
# Updates are at the bottom of the sheet
________
The data is dirty with images of varying quality and size. There are 4250 known whales represented by 9040 images, of which 2220 whales have only one(!) image in the training set. Images per whale / unique whales are distributed as follows:
1:	2220, 
2:	1034,
3:	492,
4:	192,
5:	102,
6:	61,
7:	40,
8:	23,
9:	21,
etc.
The unique wales with multiple images represent a long tail and are very few in count.

## Archtecture

Model strategies:
1. Softmax does not appear to be a good choice, as it would be a massive one and the model would need to be retrained for every new whale.
2. A model which generates whale image embeddings and then evaluates them from the standpoint of positive whale id vs negative whale id appears to be a much better choice. Also, once the model is trained, a new whale image can be added to the whale id database and the system would be expected to recognize the same whale should a new picture be presented. This is a one-shot learning problem, i.e. being able to recognize a whale from one picture only and is similar to facial recognition solution.

With the choice of model in mind, the scarcity of data appears less of a constraint and additional data can be created by manipulating  images later. As an initial step, I have made the decision to train the model on the ~2000 unique whales representing ~6800 images, without major image manipulations to create more data. This group includes whales with at least two images. Aside from convenience and not having to deal with data creation at this stage, this helps avoid the problem of one-image unique features: many pictures have unique artifacts, such as text descriptions and image idiosyncrasies which are likely to be ported to derivative images. Dealing with two different pictures of the same whale one doesn't have to deal with such artifacts. The model is built in Keras, using knowledge transfer, and works as follows:

VGG16 pre-trained model is the first segment that detect low/higher level features. The top layers of the model are disabled and substituted with custom trainable layers, together representing model whaleModel. This model is applied individually to three data feeds: Anchor, Positive and Negative. As a result, an embedding is produced for each Anchor, Positive and Negative, which are subsequently fed into a whaleModel_fuse with a custom triplet_loss objective function. The objective function seeks to maximize the difference between Anchor minus Positive, and Anchor minus Negative, and trains the top layers of whaleModel accordingly. The goal is to teach the model to recognize pictures of the same whale vs different whale, and to provide an embedding sufficiently close to that of the stored embedding to recognize a specific whale.


## Data cleaning

Data processing strategies:
1. Create more data with available image, e.g. for unique whales with only 1 image create additional images by tilting, zooming in/out, etc.
2. I decided to focus initially on the whales for which at least two images are available. This choice is due to the model choice. A success in building embedding based on the part of the set where multiple images are available would obviate the necessity to build more data.
3. I will refocus on cleaning once the model has proved effective and will possibly just work on image pairing and maybe avoid doing image manipulations altogether or at least limit its scope dramatically.

## Training

The model is currently being trained. The problem here is the speed: it takes ~5 hours to train one epoch, so it's a problem for a much bigger computing budget. 

I have succeeded in achieving a meaningful separation of the positives vs negatives on a small data set  - see results below. I limited the number of whales to 23 and examined images to 327 (all whales for which we have between 13 and 16 pictures.

Next steps: 
1. More training. There is little risk of overfitting because of the way the triple training set is designed: anchor scrolls through the available images, picks up a positive at random (if more than 1 is available) and picks a negative at random (from all available images that are not those of the anchor whale), thereby every training step is likely done on a different permutation. My goal is to extend the performance to a much larger input data set.

2. At this stage, hands-on selection of data inputs is required to fine-tune model performance. By looking at misclassified triplets of images, it is pretty clear why the algorithm made a mistake because images are quite similar. It would make sense to start pairing positives and negatives more selectively so that the compared images resemble of each other, and adding such triplets to training data. Creation of more data as appropriate might also be productive.

## Results

After 10 epochs on a very small data set just to make sure the model works, and using the data from the training set (i.e. data model has already seen), having increased alpha (distance) to 0.5:

TEST KNOWN POSITIVES, Average distance = 0.28
Out of 7: 5 identified correctly, 1 misidentified as negative and 1 misidentified from list of positives.

TEST KNOWN NEGATIVES, Average distance = 0.61	
Out of 7: 6 identified as negative, 1 misidentified as positive.

CONCLUSION: THE MODEL WORKS! Next: after an extra 50 epochs on the same set of data but additionally shuffling permutations of paired images for every epoch, got this (and make sure to compare distances for positive and negative):

TEST KNOWN POSITIVES, Average distance = 0.19
Out of 7: all identified correctly

TEST KNOWN NEGATIVES, Average distance = 0.78	
Out of 7: 6 identified as negative, 1 misidentified as positive.
CONCLUSION: IT HAS CLEARLY IMPROVED A LOT

Also tested on data the model has not seen; as expected the gaps are narrower, overlaps are larger; the model clearly needs more training on a larger set of data, however!! at this point the separation of positive and negative distances is clear, and just tested a set never seen [feb23]: 3 out of 7 positives recognized correctly, 3 not recognized, 1 recognized incorrectly. 2 out of 7 negatives misidentified as positive. CONCLUSION: I HAVE ALSO SUCCEEDED IN BUILDING EFFECTIVE EMBEDDINGS. The only significant challenge I see is computational budget. Assuming it's solved, I expect manual pairing of similar whale images to be productive in training good embeddings. I checked some misclassified images and in many cases it is pretty clear whey the model made a mistake - it would be hard for me to tell the difference.

Update Apr 23: I have trained the model on GPU and tested a sample where the model was expected to correctly identify the whale out of 2K images. I got it to ~20 percent correctly id'd. It worked very well on a small sample which served a good proof of concept. Now that GPU took care of the processing speed, the size of data is the bottleneck to be dealt with. Currently the model builds anchor-positive-negative in the following fashion: for each image, a random positive image and a random negative image are selected, with feature augmentation for each image. I will change the setup as follows: for each image, I will build all permutations of positives, and a random negative for each permutation. This should result in a much larger training set, which hopefully will be enough to improve the model performance. The problem to solve is how to feed the images from directory as they will be too large for memory. This is less straightforward than it seems despite Keras having an appropriate generator function, which works for single-stream data (I have three).

Update May 4: I have built a data generator to feed data from file with simultaneous random image augmentation taking advantage of multiple threads for image preprocessing before feeding them to GPU.    Now the training is done in a loop, where for each epoch a data set is rebuilt anew. Each data set consists of several repeated AP sets where two or three images for the same whale are available (while positives are identical and randomly transformed, negatives are random), and random selections from permutations where many images of the same whale are available, capped at a certain threshold. This way I am trying to balance the skewed nature of the data set. The model is being trained; the loss is decreasing, slower than before as regularization became stronger, and I am seeing an improvement in correctly identified whales. Depending on where it gets to, I might swap out VGG16 for a more powerful model next.





