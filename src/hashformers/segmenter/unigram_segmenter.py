from math import log10
from wordfreq import get_frequency_list

class Pdist(dict):
    """
    A probability distribution estimated from word counts
    Notice: if pw = Pdist(unigrams, n_tokens:
        * pw[w] is the raw count of the word w
        * pw(w) is the probability of the word w
    """

    @staticmethod
    def default_unk_func(key, total):
        return 1. / total

    def __init__(self, data=None, total=None, unk_func=None, **kwargs):
        super().__init__(**kwargs)

        # insert the word counts
        data = data or {}
        for key, count in data.items():
            self[key] = self.get(key, 0) + int(count)

        self.total = float(total or sum(self.values()))
        self.unk_prob = unk_func or self.default_unk_func

    def __call__(self, key):
        if key in self:
            return self[key] / self.total
        else:
            return self.unk_prob(key, self.total)


class UnigramSegmenter(object):
    """
    The Segmenter Class implements the Viterbi algorithm for word segmentation.
    Based on CH14 from the book Beautiful Data (Segaran and Hammerbacher, 2009)
    """
    def __init__(self, max_split_length=20, **kwargs):
        """
        Args:
            corpus (str): the statistics from which corpus to use for
                the spell correction.
            max_split_length (int): the maximum length of that a word can have
                for looking for splits
        """
        self.unigrams = self.read_stats(**kwargs)
        self.N = sum(self.unigrams.values())
        self.L = max_split_length

        self.Pw = Pdist(self.unigrams, self.N, self.unk_probability)

    @staticmethod
    def read_stats(lang='en', wordlist='best', match_cutoff=None):
        """
        Read key,value pairs from file.
        """
        output = {}
        frequency_list = get_frequency_list.__wrapped__(lang, wordlist=wordlist, match_cutoff=match_cutoff)
        for index in range(len(frequency_list)):
            j = ~index
            for word in frequency_list[j]:
                output[word] = index
        return output

    def condProbWord(self, word, prev, ngram_sep='_'):
        """
        Conditional probability of word, given previous word
        if bigram is not in our list, then fall back to unigrams
        Args:
            word (): candidate word
            prev (): previous observed word

        Returns:

        """
        return self.Pw(word)

    @staticmethod
    def unk_probability(key, total):
        """
        Estimate the probability of an unknown word, penalizing its length
        :param key: the word
        :param total: the count of all tokens
        :return:
        """
        return 10. / (total * 10 ** len(key))

    @staticmethod
    def combine(first, rem):
        """
        Combine first and rem results into one (probability, words) pair
        :param first: a tuple in the form: probability, word
        :param rem: a tuple in the form: probability, list_of_words
        :return:
        """
        (first_prob, first_word) = first
        (rem_prob, rem_words) = rem
        return first_prob + rem_prob, [first_word] + rem_words

    def splits(self, text):
        """
        Return a list of all possible (first, rem) pairs with max length of first <=L
        :param text:
        :return:
        """
        return [(text[:i + 1], text[i + 1:])
                for i in range(min(len(text), self.L))]

    def find_candidates(self, text, prev='<S>'):
        candidates = [
            self.combine(
                (log10(self.condProbWord(first, prev)), first), 
                self.find_segment(rem, first))
                      for first, rem in self.splits(text)]
        return candidates

    def find_segment(self, text, prev='<S>'):
        """
        Return (log P(words), words), where words is the best estimated segmentation
        :param text: the text to be segmented
        :param prev:
        :return:
        """
        if not text:
            return 0.0, []
        return max(self.find_candidates(text, prev=prev))

    def segment(self, word):
        if word.islower():
            return " ".join(self.find_segment(word)[1])
        else:
            tmp = word.lower()
            out = self.find_segment(tmp)[1]
            out_len = [ len(x) for x in out ]
            seg = []
            i = 0
            for j in out_len:
                for _ in range(j):
                    seg.append(word[i])
                    i += 1
                seg.append(" ")
            return "".join(seg).strip()
