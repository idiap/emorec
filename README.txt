############################################################################

             Emotion-Based Recommendation Generator (EMORec v1.0)      

############################################################################


README:
=======
A Python library which performs emotion-based analysis and recommendation using a 
multiple-instance regression algorithm for a set of multimedia items described by 
transcripts. The algorithm is trained over 1200 TED talks using the original human-
made transcripts and the corresponding community emotion labels. The library can be 
used in command line or directly in a Python program. It takes as input a JSON file 
which contains an array of dictionaries that describe the metadata of multimedia items 
and generates an output JSON file which contains the same items augmented with the 
following attributes:

    emotion_classes         The class names of 12 TED community emotion labels
    emotion_scores          Estimated values for 12 TED community emotion labels		
    emotion_rec             Recommended items based on these emotions	
    emotion_rec_scores      Confidence of the recommended item
    emotion_segments        Textual segments that were used
        text                The actual textual content of the segment
        start_time          Starting time of the segment
        end_time            Ending time of the segment
        relevance_scores    Relevance which reveals the contribution of the segment 
                            to the prediction of the 14 emotion dimensions.

FILES:
======
The library contains the following files:
   
    ap_weights.py     Data class for items (text extraction, preprocessing)
    crls.py           Vector space class supporting TF-IDF, LSI, RP and LDA
    generate.py       Main class responsible for generating recommendations
    data/             Data to be used for training
    models/           Pre-trained regression models on TED for emotion prediction
    parameters/       Optimal values obtained from cross-validation to be used
                      for training and prediction
           

USAGE:
======
USAGE: python generate.py -input=<path> -output=<path>
	-input	 Path location of the input file in JSON format
	-output	 Path location of the output file in JSON format

EXAMPLE:
========
$  python generate.py --input=input.json --output=output.json --debug
{'--debug': True,
 '--display': False,
 '--help': False,
 '--input': 'input.json',
 '--output': 'output.json',
 '--version': False}
[+] Loading items:....................................[OK]
[+] Modeling emotions:
        -> Unconvincing...............................[OK]
        -> Fascinating................................[OK]
        -> Persuasive.................................[OK]
        -> Ingenious..................................[OK]
        -> Longwinded.................................[OK]
        -> Funny......................................[OK]
        -> Inspiring..................................[OK]
        -> Jaw-dropping...............................[OK]
        -> Courageous.................................[OK]
        -> Beautiful..................................[OK]
        -> Confusing..................................[OK]
        -> Obnoxious..................................[OK]
[+] Generating recommendations........................[OK]
[+] Saving to output file.............................[OK]
[x] Finished.

DEPENDENCIES:
============
1) Install python: http://www.python.org/getit/
2) Install pip: http://www.pip-installer.org/en/latest/installing.html
3) Then:
$ pip install docopt
$ pip install json
$ pip install pyyaml
$ pip install numpy
$ pip install scipy
$ pip install gensim
$ pip install nltk
$ python
>>> import nltk
>>> nltk.download()

TROUBLESHOOTING:
================ 
Q: How can I use the library with items stored in other formats than JSON?
A: You have to convert your file to JSON.
Q: How can I use the library directly inside a Python program?
A: Simply import the library in Python and initialize a generator object with 
   the item dictionary of your preference.
Q: Is there any attribute that is required to be present in the item metadata?
A: Yes the 'id' attribute is mandatory.

CONTACT:
========
Nikolaos Pappas 
Idiap Research Institute
Centre du Parc, 
CH 1920 Martigny, 
Switzerland
E-mail:  nikolaos.pappas@idiap.ch 
Website: http://people.idiap.ch/npappas/ 

---
Last update:
8 Jul, 2014