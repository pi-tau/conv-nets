# DEEP CONVOLUTIONAL NETWORKS

## INCEPTION (GoogLeNet)
GoogLeNet won the ImageNet Challenge in 2014. The architecture of the model is
presented in:
  * *Going deeper with convolutions* by Szegedy et. al.,
  ([here]https://arxiv.org/abs/1409.4842)

Instead of trying to identify which convolution, ranging from `1x1` to `11x11`,
would be best, this model simply concatenates multi-branch convolutions. An
inception block applies four different convolution blocks separately on the same
input feature map: a `1x1`, `3x3`,`5x5` convolution, and a max pool operation.
Finally, it concatenates the outputs along the channel dimensions. The design
aligns with the intuition that visual information should be processed at various
scales and then aggregated so that the next stage can abstract features from
different scales simultaneously.

!["Vanilla Inception Block"](img/inception_block_vanilla.png)

One big problem with this naive form is that even a modest number of `5x5`
convolutions can be prohibitively expensive on top of a convolutional layer with
a large number of filters. This problem becomes even bigger because of the use
of a max pool layer in the mix. The max pool branch would output the same number
of filters as its input, thus the output of the inception block would inevitably
grow in filter size. For this reason `1x1` convolutions are used to compute
reductions before the expensive `3x3` and `5x5` convolutions, and also after the
max pool operation. They also include the use of a `relu` non-linearity.

!["Inception Block"](img/inception_block.png)

This networks works so well because of the combination of different filter sizes.
They explore the image on different scales, meaning that details at different
extents can be recognized efficiently by the different filters. The commonly
tuned hyper-parameters of the Inception block are the number of output channels
for each of the branches, i.e., how to allocate capacity along the convolutions
of different size. The main intuition is to have the most filters for the `3x3`
convolutions as they are powerful enough to take the context into account while
requiring almost a third of the parameters of the `5x5` convolution.

The GoogLeNet architecture consists of stacking multiple Inception blocks with
occasional max pooling to reduce the height and the width of the feature maps.
It exhibits a clear distinction among the stem (data ingest), body (data
processing), and head (prediction). This design pattern has persisted ever since
in the design of deep networks.

!["GoogLeNet"](img/googlenet.png)

The original design also included auxiliary classifiers (small neural nets)
connected to intermediate layers (first and fifth layer of the five layer block)
in order to stabilize training. By adding these auxiliary classifiers,
the authors state that they increase the gradient signal that gets propagated
back and also provide additional regularization. During training their loss gets
added to the total loss of the network with a discount weight (`w=0.3`). These
tricks to stabilize training are no longer necessary due to the availability of
improved algorithms.

