#python 3.6

# !pip install --upgrade tables > temp.txt
import random
import pandas as pd
from torch.utils import data
import torch
import gensim
import re
from emoji import UNICODE_EMOJI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from torch.utils import data


def cleanText(tl):
	'''
	removes URL
	removes repeating question marks
	removes emojis
	'''
	emojis = ''.join((UNICODE_EMOJI).keys())
	for i in range(len(tl)):
		tl[i] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',tl[i]) # <---- removes URL
		tl[i] = re.sub(r'\?\?+','',tl[i]) # <---- removes repeating question marks
		tl[i] = tl[i].replace('�','') # special case
		tl[i] = tl[i].translate(str.maketrans('','',emojis)) # <---- remove emojis
		tl[i] = re.sub(r'\s+',' ',tl[i]) # remove repeating spaces
	return tl

def rmstp(tl):
	'''
	remove stopwords

	'''
	# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
	stop_words = set(stopwords.words('english'))

	for i in range(len(tl)):
		word_tokens = word_tokenize(tl[i]) 
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		tl[i] = ' '.join(filtered_sentence)

	return tl

def rtPOS(tl, pos=['']):
	'''
	retain only the given POS tags
	'''
	for i in range(len(tl)):
		word_tokens = word_tokenize(tl[i])
		tagged = nltk.pos_tag(word_tokens) 
		s = ''
		for t in tagged:
			if t[1] in pos:
				s = s + ' ' + t[0]
				s = s.strip()
		tl[i] = s
	return tl

def rmhashtags(tl):
	'''
	remove hashtags
	'''
	for i in range(len(tl)):
		tl[i] = re.sub(r'#.+?\b','', tl[i])
		tl[i] = re.sub(r'\s+',' ',tl[i]) # remove repeating spaces
	return tl

def rmhashes(tl):
	'''
	just remove hash symbols
	'''
	for i in range(len(tl)):
		tl[i] = tl[i].replace('#','')
	return tl

def rmmentions(tl):
	'''
	removing the @ mentions
	'''
	for i in range(len(tl)):
		tl[i] = re.sub(r'@.+?\b','', tl[i])
		tl[i] = re.sub(r'\s+',' ',tl[i]) # remove repeating spaces
	return tl

def rmpunct(tl):
	'''
	removes all punctuations
	'''
	for i in range(len(tl)):
		tl[i] = tl[i].translate(str.maketrans('', '', string.punctuation))
		tl[i] = re.sub(r'\s+',' ',tl[i]) # remove repeating spaces
	return tl

def trim(tl, n):
	'''
	trim the give sentences to a given length
	'''
	for i in range(len(tl)):
		word_tokens = word_tokenize(tl[i])
		word_tokens = word_tokens[:n]
		tl[i] = ' '.join(word_tokens)
	return tl

def rmFact(tl):
	'''
	removes Fact: string
	'''
	for i in range(len(tl)):
		if "Fact:" in tl[i]:
			tl[i] = tl[i][:tl[i].index("Fact:")]
		if "Fact :" in tl[i]:
			tl[i] = tl[i][:tl[i].index("Fact :")]
	return tl


class mydataset(data.Dataset):
	def __init__(self, seed=123, embeddings=False, removeStopwords=True, removeHashtags=False, removeHashes=True, removePunct=False, max_length=0, removeMentions=True, SentSeg=False, Factremoval=True, pos=[]):
		self.seed=seed
		self.removeStopwords=removeStopwords
		self.removeHashtags=removeHashtags
		self.removeHashes=removeHashes
		self.removePunct=removePunct
		self.max_length=max_length
		self.removeMentions=removeMentions
		self.Factremoval=Factremoval
		self.doEmbeddings=embeddings
		self.pos=pos
		df = pd.read_csv('data/Constraint_English_Train - Sheet1.csv')
		text = list(df['tweet'])
		y = [1 if x == 'real' else 0 for x in list(df['label']) ]
		text = cleanText(text)

		# Dont change the order of these if blocks here

		if self.removeHashtags:
			text = rmhashtags(text)

		if self.removeHashes:
			text = rmhashes(text)

		if self.removeMentions:
			text = rmmentions(text)

		if self.Factremoval:
			text = rmFact(text)

		if self.removeStopwords:
			text = rmstp(text)

		if self.pos:
			text = rtPOS(text, self.pos)

		if self.removePunct:
			text = rmpunct(text)

		if self.max_length:
			text = trim(text, self.max_length)

		self.text = text
		self.y = y

	
	def __getitem__(self, index):
		return self.text[index], self.y[index]

	def __len__(self):
		return len(self.y)

if __name__ == '__main__':
	

	dataset = mydataset()
	train_set, val_set, test_set = torch.utils.data.random_split(dataset,[int(len(dataset)*0.7), int(len(dataset)*0.1), len(dataset)-int(len(dataset)*0.8)])
	train_data = data.DataLoader(train_set, batch_size=32, shuffle=False, num_workers=4)
	val_data = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
	test_data = data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

	for i, (text, y) in enumerate(train_data):
		for i in range(len(y)):
			print(text[i], y[i])
		break


