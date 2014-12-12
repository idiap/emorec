#    Copyright (c) 2014 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of EMORec.
#
#    EMORec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    EMORec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar. If not, see <http://www.gnu.org/licenses/>.

"""Usage:
    generate.py --input=<path> --output=<path> [options]

Options:
    -v, --version                      show program's version number and exit
    -h, --help                         show this help message and exit
    -d, --debug	                       print status and debug messages [default: False]
    -r, --display                      display recommendations per item [default: False]
    -i, --input=<path>                 path to JSON file to be used as input
    -o, --output=<path>                path to JSON file to be used as output
"""
import sys
import json
import gzip 
import numpy as np
import cPickle as pickle
from math import sqrt
from scipy import linalg 
from utils import Unbuffered, write
from docopt import docopt
from itertools import izip 
from ap_weights import APWeights
from scipy.sparse import lil_matrix
from nltk.tokenize import RegexpTokenizer


class Generator:
	def __init__(self, options, items=None):
		data_path = "data/ted_transcript_5k.p"
		data = pickle.load(open(data_path))
		self.dictionary = pickle.load(open(data_path.replace(".p",".dict")))
		self.tfidf = pickle.load(open(data_path.replace(".p",".tfidf")))
		self.X = data["X"]
		self.Y = data["Y"]
		self.items = items  
		self.words = data["words"]
		self.rating_classes = data["classes"]
		self.debug = options['--debug']	
		self.display = options['--display']		
		self.input_path = options['--input']
		self.output_path = options['--output']
		self.lim = 140 # maximum number of words per chunk

	def run(self):

		emotion_scores = []
		emotion_rec = []
		relevances = []

		# Loading test data and representing them in the feature space.
		write("[+] Loading items:".ljust(54,'.')) if self.debug else ''
		if self.items is None: 
			self.items = json.loads(open(self.input_path).read()) 
		test_data, starts, ends = self.get_text() 
		X_test = []
		count = 0
		for i, sentences in enumerate(test_data):
			emotion_rec.append([])
			relevances.append([])
			emotion_scores.append([0. for k in range(12)])
			cur_x = lil_matrix((len(sentences), len(self.dictionary.keys())))
			for j, sentence in enumerate(sentences):
				relevances[i].append([0. for k in range(12)]) 
				sword_vector = self.vectorize_document(sentence)
				sfeature_vector = self.feature_extraction(sword_vector)
				for key, val in sfeature_vector:
					cur_x[j,key] = val 
				count += 1
			X_test.append(cur_x)
		write("[OK]\n") if self.debug else ''

		# Training and predicting each emotion class on the given data.
		write("[+] Modeling emotions:") if self.debug else ''
		for num_class in range(12):
			self.num_class = num_class + 1
			cur_X = self.X
			cur_Y = self.Y[:,(self.num_class - 1):self.num_class] 
			best = None
			dataset_name = "ted_talks"
			method = "ap_weights"
			rating_class = self.rating_classes[self.num_class-1]
			opt = []
			for b in ['_m1','','_m3']:
				tmp = open('parameters/%d/%s%s.txt' % (self.num_class,method,b))
				best = float(tmp.readlines()[0].split(": ")[1].split("}")[0])
				opt.append(best)
			try:
				write(("\n" + " "*8 + "-> %s"  % rating_class).ljust(55,'.')) if self.debug else ''
				f = gzip.open("models/%s_model.pk" % rating_class, 'rb')
				self.model = pickle.load(f) 
				write("[OK]") if self.debug else ''
			except:
				write("\n") if self.debug else ''
				self.model = APWeights(20, l1=opt[0], l2=opt[1], l3=opt[2],  reg=self)	 
				self.model.fit(cur_X, cur_Y)  
				f = gzip.open("models/%s_model.pk" % rating_class,"wb")
				pickle.dump(self.model, f)

			pred = self.model.predict(X_test)
			for j, val in enumerate(pred):
				emotion_scores[j][num_class] = val.view(np.ndarray)[0]

			for j, weights in enumerate(self.model.P_test): 
				for i, w in enumerate(weights):
					relevances[j][i][num_class] = w

		# Compute emotion-based recommendations
		write("\n[+] Generating recommendations".ljust(55,'.')) if self.debug else ''
		sim = np.zeros((len(X_test), len(X_test)))
		for i, v1 in enumerate(emotion_scores):
			for j, v2 in enumerate(emotion_scores):
				sim[i][j] = self.cosine_measure(v1, v2)
		write("[OK]") if self.debug else ''

		real_idxs = [nid for nid in range(len(X_test))]

		# Write results into a file
		write("\n[+] Saving to output file".ljust(55,'.')) if self.debug else ''
		output = self.items
		for j, h in enumerate(self.items):
			segments = test_data[j]
			h['segments'] = []
			for i,seg in enumerate(segments):
				seg_h = {'text':seg, 
						 'start':starts[j][i],
						 'end':ends[j][i],
						 'relevance_scores':relevances[j][i]}
				h['segments'].append(seg_h)
			top, top_sim = self.n_most_similar(j, sim, real_idxs)
			h['emotion_classes'] = self.rating_classes
			h['emotion_scores'] = emotion_scores[j]
			h['emotion_rec'] = [self.items[idx]['id'] for idx in top]
			h['emotion_rec_scores'] = top_sim

		json.dump(output, open(self.output_path,"wb"))
		write("[OK]\n") if self.debug else ''
		print "[x] Finished."
	
	def get_text(self):
		all_chunks = []
		all_starts = []
		all_ends = []
		for i, talk in enumerate(self.items):
			all_chunks.append([])
			all_starts.append([])
			all_ends.append([])			
			chunk = []
			start = None
			end = None
			count = 0
			for seg in talk['segments']:
				if seg['classId'] == "speech":
					for word in seg['spokenWords']:
						if start is None:
							start = word['wordStart']
						else:
							end = word['wordEnd']
						chunk.append(word['wordId'])
						count += 1
				if count > self.lim:
					all_chunks[i].append(' '.join(chunk))
					all_starts[i].append(start)
					all_ends[i].append(end)
					chunk = []
					start = None
					end = None
					count = 0
		return all_chunks, all_starts, all_ends
	
	def vectorize_document(self, document):
		tokenizer = RegexpTokenizer(r'\b[A-z]+\b')
		words =  list(tokenizer.tokenize(document.lower()))
		return words

	def feature_extraction(self, vector_document):
		sparse_vector = self.dictionary.doc2bow(vector_document)
		sparse_vector = self.tfidf[sparse_vector]
		return sparse_vector

	def cosine_measure(self, v1, v2):
	    return (lambda (x, y, z): x / sqrt(y * z))(reduce(lambda x, y: (x[0] + \
	    		y[0] * y[1], x[1] + y[0]**2, x[2] + y[1]**2), izip(v1, v2), (0, 0, 0)))

	def n_most_similar(self, cur_idx, sim, real_idxs, N=10):
		similar = []
		confidence = []
		for idx in np.argsort(sim[cur_idx])[::-1][:N]:
			if idx != cur_idx:
				similar.append(real_idxs[idx])
				confidence.append(sim[cur_idx][idx])
		return similar, confidence


if __name__ == '__main__': 
    options = docopt(__doc__, version='EMORec v1.0, NLP Group @ Idiap 2014')
    print options
    gen = Generator(options, items=None)
    gen.run()
