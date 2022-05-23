# Tort
<b>Tort</b>: Vi<b>T</b> for m<b>ort</b>als

# Abstract
Recently, models free of prior knowledge of the data have been gaining
popularity, and they owe this to the opportunity for people to utilize huge
clusters for machine learning tasks and use huge size datasets. Without this
it is almost impossible to train such a model that can not make assumptions
about the data it is receiving. One of such a models is Vision Transformer. We
stated our goal as optimizing the Vision Transformer training process to reduce
required computations and data volume. Analysing this problem we understood
that the signal from supervised objective may be not enough and we added
several self-supervised tasks for the model so it can obtain more information
every backward pass during training. Also we manipulate input batch so it
provide more details keeping memory footprint almost the same. We compared
our method of training ViT to existing state-of-the-art recipe of training ViT and
report valuable advances, including twice faster speed of the training, significant
classification metrics improvements and accelerated convergence.

# Installation and dependencies
Look into `REQUIREMENTS.md`
