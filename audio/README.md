# Audio Generation
This code is an attempt to generate audio from a raw PCM stream using:

* LSTM using relu activation
* Markov chains

## Ideas
* Discretize the input stream into octaves
* Use FFT/IFFT to learn the frequencies wrt amplitude, generate
* Use nupic
## Resources
Some resources I found:

* Reddit Discussion:
  * https://www.reddit.com/r/MachineLearning/comments/31xufa/markov_chain_audio_generation/
* Using Conv nets:
  * https://dl.dropboxusercontent.com/u/19706734/paper_pt.pdf
* Markov-chain monte-carlo for audio segmentation
  * http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1661396&tag=1