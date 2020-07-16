# Ideas

Any ideas you want to have written in the repo.

## Maddie's Ideas

Start off with a simple shallow network (one layer) pretty readily available and simple to make
(perhaps I could make an test it as a complete beginner just to show like the most simple base that we are then going to build on from with different/novel
techniques in order to try and make a more efficient and/or accurate network. Our first foray into different techniques should probably include looking
at a deep network - seeing how many more layers we can add to the network and what impact this has on the accuracy and speed (predicting slower but more
accurate but who knows! In this case i’m not even sure a deep network would be better? like sometimes less is more and I really feel like this is one of
those situations.)

I think investigating recurrent nets is probably the most adventurous direction we can go in that he suggested. Barely understanding neural networks at all,
the concept of a recurrent net is a bit difficult for me to understand however I do think it is worth looking in to or at least mentioning in the presentation
with regards to our thoughts on it and how it would relate to the problem. A benefit of the recurrent nets is that they can be used to simply solve problems
that are difficult to solve with feedforward nets. Since this isn’t really an issue with MNIST classification (it’s relatively simple to do with a feedforward net)
it’s probably worth noting that while exploring this will be interesting it might not be fruitful (still, worth doing it for the points i think, and if
we can actually pull it off and make one that is comparable to a feedforward net we’ll be pretty well placed I think). Can also note that recurrent nets,
while currently running on less powerful algorithms, are structurally more similar to way that real neutrons fire in the brain (and getting artificial
intelligence to imitate real intelligence is general goal right?).

Then we can also dive into CNN and say that it is intended for greater accuracy
(which isn’t strictly necessary for MNIST) but uses more power? And it’s fundamental difference is the way it breaks down the raw image data (I think).
I still don’t know what hyper parameters r tbh so i’m not sure how to compare and contrast those…
For non neural techniques we can try randomly guessing (which would have obviously a 1/10 chance of classifying the digits correctly),
guessing based on how dark the image is or perhaps splitting up the image into halves (vertically) or quarters and doing a test for darkness by sections.
Big alternative is support vector machine (SVM) (https://peekaboo-vision.blogspot.com/2010/09/mnist-for-ever.html).
Perhaps for brownie points we could also touch on the limitations of MNIST as standard benchmark for machine learning
(simplicity etc. - like it’s good for beginners but is it really for testing complex networks, seeing as such simple networks can achieve a high
degree of accuracy / efficiency relatively easily.) Also Will doesn’t seem like he minds having the whole basis of his project questions and wld
probs give us credit for considering talking about it. Perhaps we could look at other datasets (maybe numbers written by people who don’t natively use the
roman number system (i.e. Japanese or Arabic native speakers) to see how the network would hold up classifying numbers written by foreigners (like MNIST
has a pretty shallow sample group - besides the fact they’re all american anyway). 