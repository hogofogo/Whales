# whales_model


https://www.kaggle.com/c/whale-categorization-playground

"This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers 
and given an Id. The challenge is to predict the whale Id of images in the test set. What makes this such a challenge is that 
there are only a few examples for each of 3,000+ whale Ids."
=======
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

Data processing strategies:
1. Create more data with available image, e.g. for unique whales with only 1 image create additional images by tilting, zooming in/out, etc.
2. I decided to focus initially on the whales for which at least two images are available. This choice is due to the model choice - see below.

Model strategies:
1. Softmax does not appear to be a good choice, as it would be a massive one and the model would need to be retrained for every new whale.
2. A model which generates whale image embeddings and then evaluates them from the standpoint of positive whale id vs negative whale id appears to be a much better choice. Also, once the model is trained, a new whale image can be added to the whale id database and the system would be expected to recognize the same whale should a new picture be presented. This is a one-shot learning problem, i.e. being able to recognize a whale from one picture only and is similar to facial recognition solution.

With the choice of model in mind, the scarcity of data appears less of a constraint and additional data can be created by manipulating  images later.

As an initial step, I have made the decision to train the model on the ~2000 unique whales representing ~6800 images, without major image manipulations to create more data. This group includes whales with at least two images. Aside from convenience and not having to deal with data creation at this stage, this helps avoid the problem of one-image unique features: many pictures have unique artifacts, such as text descriptions and image idiosyncrasies which are likely to be ported to derivative images. Dealing with two different pictures of the same whale one doesn't have to deal with such artifacts. The model is built in Keras, using knowledge transfer, and works as follows:

VGG16 pre-trained model is the first segment that detect low/higher level features. The top layers of the model are disabled and substituted with custom trainable layers, together representing model whaleModel. This model is applied individually to three data feeds: Anchor, Positive and Negative. As a result, an embedding is produced for each Anchor, Positive and Negative, which are subsequently fed into a whaleModel_fuse with a custom triplet_loss objective function. The objective function seeks to maximize the difference between Anchor minus Positive, and Anchor minus Negative, and trains the top layers of whaleModel accordingly. The goal is to teach the model to recognize pictures of the same whale vs different whale, and to provide an embedding sufficiently close to that of the stored embedding to recognize a specific whale.

The model is currently being trained. The problem here is the speed: it takes ~5 hours to train one epoch, so it's a problem for a much bigger computing budget. Assuming the training reaches the point where a meaningful separation of the positives vs negatives is achieved:

Next steps: 
1. More training. There is little risk of overfitting because of the way the triple training set is designed: anchor scrolls through the available images, picks up a positive at random (if more than 1 is available) and picks a negative at random (from all available images that are not those of the anchor whale), thereby every training step is likely done on a different permutation.
2. At this stage, hands-on selection of data inputs is required. By looking at misclassified triplets of images, it is pretty clear why the algorithm made a mistake because images are quite similar. It would make sense to start pairing positives and negatives more selectively so that the compared images resemble of each other, and adding such triplets to training data. Creation of more data as appropriate might also be productive.
