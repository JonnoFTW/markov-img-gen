# Markov Chain Image Generation

This is a small script that generates images from other images using a markov chain.

##Dependencies

You will need:

* Python 3
* numpy
* pillow
* requests
* scipy
* pyprind

If you want to try the audio generation, you will need:

* scipy
* keras
* matplotlib
* sounddevice

## Usage

To use the image generator, you can provide either a local file or remote url:

```
usage: imggen.py [-h] -i INPUT [-b BUCKETS] [-ow WIDTH] [-oh HEIGHT] [-n] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Image to learn from. Can be a local file or url
  -b BUCKETS, --buckets BUCKETS
                        Training bucket width
  -ow WIDTH, --width WIDTH
                        Width of output image
  -oh HEIGHT, --height HEIGHT
                        Height of output image
  -n, --eight-neighbours
                        Train on all 8 neighbours, default is 4
  -d, --directional     Train the image using the relative location of each
                        neighbour
  -s, --show-normalized Show the normalized (just apply the bucketing) image only

```
The most basic example is:
```
./imggen.py -i img.jpg
```

Here is an example using an image url with 720x720 output, buckets of width 8, train on all 8 neighbours and learn the direction of colours:
```
./imggen.py -i https://i.imgur.com/Er2xlip.jpg -oh 720 -ow 720 -b 8 -n -d
```

## Examples

![HomerInput](http://i.imgur.com/ql46UZL.png)
![HomerGenerated](http://i.imgur.com/QDqCIF9.png)

An image generated from my black and white portrait:

![MeGenerated](http://i.imgur.com/RtyeYAJ.png)

## Audio Generation

These scripts are incomplete attempts at generating audio using a markov chain from a raw PCM stream. The best I could get was a constant tone, which beat the previous outputs of what was essentially noise.