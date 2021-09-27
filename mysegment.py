
import io
import math
import os.path as op
import sys


class Segmenter(object):
    """Segmenter

    Support for object-oriented programming and customization.

    """
    ALPHABET = set('abcdefghijklmnopqrstuvwxyz')

    def __init__(self,vocab_file="subword_dataset/vocab",limit=12):
        self.unigrams = {}
        self.total = 0.0
        self.limit = limit
        self.UNIGRAMS_FILENAME = vocab_file
        self.load()

    def load(self):
        "Load unigram and bigram counts from disk."
        self.unigrams.update(self.parse(self.UNIGRAMS_FILENAME))
        with open(self.UNIGRAMS_FILENAME,"r") as f:
            total = int(f.readline().strip().split()[1])
        self.total = total

    @staticmethod
    def parse(filename):
        "Read `filename` and parse tab-separated file of word and count pairs."
        with io.open(filename, encoding='utf-8') as reader:
            lines = (line.split('\t') for line in reader if not line.startswith("<total>"))
            return dict((word, float(number)) for word, number in lines)

    def score(self, word):
        "Score `word` in the context of `previous` word."
        unigrams = self.unigrams
        total = self.total

        if word in unigrams:

            # Probability of the given word.

            return unigrams[word] / total

        return 0

    def best_segment(self, text):
        "Return iterator of words that is the best segmenation of `text`."
        memo = dict()


        def search(text, previous='<s>'):
            "Return max of candidates   `text` given `previous` word."
            if text == '':
                return 0.0, []

            def candidates():
                "Generator of (score, words) pairs for all divisions of text."
                for prefix, suffix in self.divide(text):
                    score = self.score(prefix)
                    if score==0:
                        prefix_score = -99999999999999999
                    else:
                        prefix_score = math.log10(score)

                    pair = (suffix, prefix)
                    if pair not in memo:
                        memo[pair] = search(suffix, prefix)
                    suffix_score, suffix_words = memo[pair]

                    yield (prefix_score + suffix_score, [prefix] + suffix_words)

            return max(candidates())

        # Avoid recursion limit issues by dividing text into chunks, segmenting
        # those chunks and combining the results together. Chunks may divide
        # words in the middle so prefix chunks with the last five words of the
        # previous result.

        clean_text = self.clean(text)
        size = 250
        prefix = ''

        for offset in range(0, len(clean_text), size):
            chunk = clean_text[offset:(offset + size)]
            _, chunk_words = search(prefix + chunk)
            prefix = ''.join(chunk_words[-5:])
            del chunk_words[-5:]
            for word in chunk_words:
                yield word

        _, prefix_words = search(prefix)

        for word in prefix_words:
            yield word

    def multi_segment(self, text,k):
        "Return iterator of words that is the k best segmenations of `text`."
        memo = dict()
        self.iters = 0
        #if text in self.unigrams:
        #    return text

        def search(text):
            "Return max of candidates matching `text` given `previous` word."
            if text == '':
                return 0.0, []
            self.iters += 1
            def candidates():
                "Generator of (score, words) pairs for all divisions of text."
                for prefix, suffix in self.divide(text):
                    score = self.score(prefix)
                    if score == 0:
                        prefix_score = -99999999999999999
                    else:
                        prefix_score = math.log10(score)

                    pair = (suffix, prefix)
                    if pair not in memo:
                        memo[pair] = search(suffix)
                    suffix_score, suffix_words = memo[pair]

                    yield (prefix_score + suffix_score, [prefix] + suffix_words)

            if self.iters==1:
                return sorted(candidates(),key=lambda x:x[0],reverse=True)[:k]
            else:
                return max(candidates())

        # Avoid recursion limit issues by dividing text into chunks, segmenting
        # those chunks and combining the results together. Chunks may divide
        # words in the middle so prefix chunks with the last five words of the
        # previous result.

        clean_text = self.clean(text)

        words = search(clean_text)
        for word in words:
            yield word

    def segment(self, text,multi=1):
        "Return list of words that is the best segmenation of `text`."
        if multi == 1:
            return list(self.best_segment(text))
        else:
            return list(self.multi_segment(text,multi))


    def divide(self, text):
        "Yield `(prefix, suffix)` pairs from `text`."
        for pos in range(1, min(len(text), self.limit) + 1):
            yield (text[:pos], text[pos:])


    @classmethod
    def clean(cls, text):
        "Return `text` lower-cased with non-alphanumeric characters removed."
        alphabet = cls.ALPHABET
        text_lower = text.lower()
        letters = (letter for letter in text_lower if letter in alphabet)
        return ''.join(letters)

