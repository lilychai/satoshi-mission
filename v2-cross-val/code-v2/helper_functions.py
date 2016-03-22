"""
:
: Helper functions for cleaning text, creating samples, re-sampling and extracting
: features -- reusable functions whose inputs/outputs do not rely on databases.
:
"""

from __future__ import division         # no need to worry about integer division

from nltk.data import load              # to load pre-trained Punkt tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords       # using this set of stopwords for the model
import enchant                          # pyenchant -- US vs. GB spelling

from numpy.random import choice

from operator import itemgetter
from collections import Counter
from itertools import chain, izip
import string, re

import pandas as pd
import numpy as np



def bag_of_sentences(contents):
    """
    :type contents: List[str]     -- list of docs
    :rtype: List[str]             -- list of raw sentences
    :
    : Join all lines in contents and return a list of sentences.
    : (Note: "line" == split by breaks in pdf files != "sentence")
    :
    """

    sentences = []

    for doc in contents:

        doc.encode('ascii')

        # simple data cleaning to remove artifacts/equations/text
        # from figures/flowcharts

        # line break characters
        # (string.whitespace = '\t', '\n', '\x0b', '\x0c', '\r', ' ')
        # \x0c = form feed; forces a printer to move to the next sheet of paper.
        breaks = tuple(ch for ch in tuple(string.whitespace)[1:-1] \
                           if doc.count(ch) > 0)


        # remove artifact from justified text alignment
        # and change four spaces to '\t' (tab)
        doc = re.sub('    (?=\w)', '\t', doc.replace('\n         ', '\n'))


        # remove computer code variable/function names e.g. getParam, get_param
        doc = re.sub('\w+[a-z]([A-Z]|\_\w)\w+', '', doc)


        # drop references if exist (find() returns -1 if not found)
        i = doc.find('References') * doc.find('Bibliography')
        if i < 0:
            doc = doc[:-i]


        # split on line break characters; drop empty sentences
        sents = re.split('|'.join(breaks), doc)


        # remove leading/trailing ' '
        # (note: can't leave strip() empty otherwise \t is stripped as well!)
        # remove page number
        # remove empty strings ''
        # sentences with too many white characters in it (text from flowcharts)
        # drop lines with more digits than alphabets -- most likely an equation
        sents = [s.strip(' ') for s in sents \
                               if s \
                              and not s.isdigit() \
                              and 0 < s.count(' ')/len(s) < 0.2 \
                              and len(filter(string.letters.__contains__,s)) /\
                                  len(s) > 0.5]


        sentences += sents


    # join all sentences from same writine type into one long doc
    longdoc = '\v'.join(strip_misc_periods(s) for s in sentences)
    sent_detector = load('tokenizers/punkt/english.pickle')
    punkt_sentences = sent_detector.tokenize(longdoc)


    # want to retain the spaces behind periods
    bag_of_sents = []
    for sentence in punkt_sentences[:-1]:
        i = longdoc.find(sentence[-3:]) + 3  # index after the end of sentence
        longdoc = longdoc[i:]

        # find no. of spaces leading the next sentence
        delta = len(longdoc) - len(longdoc.lstrip(' '))

        # append spaces to end of sentence
        bag_of_sents += sentence + ' '*delta,

    bag_of_sents += punkt_sentences[-1],

    return bag_of_sents



def strip_misc_periods(text):
    """
    :type text: str
    :rtype: str
    :
    : Remove miscellaneous dots that are not terminal punctuation.
    :
    """

    # remove urls and emails
    text = re.sub('([\w\.]*(@|:((//)|(\\\\)))' + \
                  '[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*(?=$|[\.\s]))', '', text)

    # replace '...' and '....' with '.' (in case there are any)
    text = re.sub('(\.\.+)', '.', text)

    # remove periods preceeding/following numbers
    # (either part of a decimal number or from a bullet point)
    text = re.sub('((?<=\d)\.)|(\.(?=\d))', '', text)

    return text



def create_chunks(sentences, chunk_size=500, target_count=0, verbose=False):
    """
    :type sentences: List[str]
    :type chunk_size: int
    :type target_count: int
    :type verbose: bool
    :rtype: List[List[str]]
    :
    : Bag sentences into chunks (each chunk is an "observation" for the model),
    : keeping no. of words in each chunk roughly equal without chopping off
    : sentences.
    : (Note: word count = quick count of words in raw sentences).
    :
    : Boostrap sentences to achieve target bag count.
    :
    """

    chunks = []

    # use all sentences once
    chunk, current_remaining = [], chunk_size
    for sentence in sentences:
        new_remaining = chunk_decision_tree(sentence,
                                            current_remaining,
                                            chunk_size)

        if new_remaining <= 0:  # chunk full after append line
            chunks += chunk + [sentence],
            chunk, current_remaining = [], chunk_size

        elif new_remaining < current_remaining:  # chunk not full
            chunk += sentence,
            current_remaining = new_remaining

        else:  # chunk full -- start new chunk
            chunks += chunk,
            chunk, current_remaining = [sentence], new_remaining


    # bootstrap-fill the final chunk, then bootstrap on the sentence level
    # to create additional chunks (re-sampling)
    inds = range(len(sentences))
    flag = True                                  # fake a do-while loop
    while flag or len(chunks) < target_count:
        while current_remaining > 0:
            sentence = sentences[choice(inds)]
            new_remaining = chunk_decision_tree(sentence,
                                                current_remaining,
                                                chunk_size)
            if 0 <= new_remaining < current_remaining:
                chunk += sentence,
                current_remaining = new_remaining
            else:
                chunks += chunk,
                if flag and verbose:
                    print '    No. of chunks before boostrapping:', len(chunks)
                    print '    Sentences per chunk:', \
                          [len(chunk) for chunk in chunks]

                flag = False
                break
        chunk, current_remaining = [], chunk_size

    if verbose:
        print '    No. of chunks after boostrapping:', len(chunks)
        print '    Sentences per chunk:', [len(chunk) for chunk in chunks]

    return chunks



def chunk_decision_tree(sentence, current_remaining, chunk_size):
    """
    :type sentence: str
    :type current_remaining: int
    :type chunk_size: int
    :rtype: int
    :
    : Decide whether sentence belongs to current chunk.
    : Return remaining word availability after adding the sentence
    : to the current or a new chunk.
    :
    """

    num_words = len(word_tokenize(sentence))

    # if chunk not yet full or full after append
    if (num_words <= current_remaining) \
    or (abs(current_remaining - num_words) < current_remaining):

       return current_remaining - num_words

    # chunk full -- start new chunk
    return chunk_size - num_words




def crunch_statistics(data, tagger=None):
    """
    :type data: List[str]
    :type tagger: spacy.en.English tagger
    :rtype: Counter, List[str], List[str], List[str], List[str], List[str]
    :
    : Parse, clean and compute stylometric statistics in lines of a single doc.
    :
    """

    ctr = Counter()


    # period followed by two white spaces
    ctr['periods_twospaces'] = sum(line.count('.  ') for line in data)


    # period followed by one white spaces
    # (note: this includes periods in e.g. "Mr. ")
    ctr['periods_onespace']  = sum(line.count('. ') for line in data) - \
                               sum(ctr.values())   # avoid double counting


    # period so happened to be at the end of a line
    # (note: this includes periods in abbreviations e.g. "e.g.")
    ctr['periods_nospace']   = sum(line.count('.') for line in data) - \
                               sum(ctr.values())   # avoid double counting


    # no. of paragraphs
    ctr['paragraphs'] = sum(1 for s in data if re.search('^\t(?=[a-zA-Z])', s))


    # some things are easier to count when lines are joined
    doc = ' '.join(data)


    # no. of hyphens (e.g. in "peer-to-peer")
    ctr['hyphens'] = sum(1 for hyphen in re.finditer('(\w-)|(-\w)', doc))

    # no. of apostrophe (e.g. in "they'll")
    ctr['apostrophes'] = sum(1 for apostrophe \
                             in re.finditer("[a-z]\'[a-z]", doc))

    # no. of sentences starting with capital letter vs. lowercase
    ctr['sentences_upper'] = sum(1 for cap \
                                 in re.finditer('(^|(?<=\.))\s+(?=[A-Z])', doc))
    ctr['sentences_lower'] = sum(1 for cap \
                                 in re.finditer('(^|(?<=\.))\s+(?=[a-z])', doc))

    # nltk pre-trained sentence detector (for sanity check)
    sent_detector = load('tokenizers/punkt/english.pickle')
    ctr['sentences_punkt'] = len(sent_detector.tokenize(doc))



    # change shorthand to full word
    doc = re.sub('\'ll', ' will', doc)
    doc = re.sub('\'ve', ' have' , doc)
    doc = re.sub('n\'t', ' not' , doc)
    # shorthand "'s" and possession "'s" hard to distinguish so ignore


    # remove ascii control codes, numbers and punctuations by splitting on them,
    # lowercase all words,
    # remove empty words (''),
    # remove single characters not 'a' or 'I' -- from equations or annotations,
    # remove author names (limited to satoshi and suspects)
    words = np.array([word for word in \
                      re.split(r'[^\x00-\x7F]|[' + string.whitespace + \
                                                   string.punctuation + \
                                                   string.digits + ']',
                               doc.lower()) \
                      if word \
                     and (len(word) > 1 or word in {'a', 'i', 'A', 'I'}) \
                     and word not in {'satoshi', 'nakamoto',
                                      'suspectA-firstname', 'suspectA-lastname',
                                      'suspectB-firstname', 'suspectB-lastname',
                                      'suspectC-firstname', 'suspectC-lastname',
                                      'francisco', 'cervera'}])


    # no. of words incl. stop words
    ctr['words'] = len(words)


    # no. of words in US english (dictionaries are case sensitive!)
    d1 = enchant.Dict('en_US')
    isUS = np.array([is_US(word) for word in words])
    ctr['words_us'] = sum(isUS)


    # no. of words in GB english
    d2 = enchant.Dict('en_GB')
    isGB = np.array([is_GB(word) for word in words])
    ctr['words_gb'] = sum(isGB)



    # add an if check for satoshi forum@470: empty words array
    misspellings, gb_spellings, us_spellings = set(), set(), set()
    if len(isUS) > 0 < len(isGB):

        misspellings = words[~isUS & ~isGB]     # non-english words

        # in case a word is split into two lines by hyphen
        joins = [t1+t2 for t1,t2 in zip(misspellings[:-1], misspellings[1:])]
        for word in joins:
            check = (is_US(word), is_GB(word))
            if any(check):
                words = np.append(words, word)
                ctr['words_us'] -= 2 * check[0]
                ctr['words_gb'] -= 2 * check[1]
                isUS = np.append(isUS, True if check[0] else False)
                isGB = np.append(isGB, True if check[1] else False)


        # save spellings lists for visual inspection
        gb_spellings = set(words[~isUS]) - set(words[~isGB])  # not in US = GB
        us_spellings = set(words[~isGB]) - set(words[~isUS])  # not in GB = US


        # use only words that are in the US/GB dictionaries,
        # so that words improperly cleaned won't create noise
        tokens = [w for w in tagger(u' '.join(words[isUS|isGB])) \
                     if w.pos_ not in {u'X', u'PUNCT'}]

        # if using the coarse-grained, less detailed tag,
        # exclude words classified as X or PUNCT
        # (this removes names, latin words, and a few misclassified nouns)
        pos_ctr = Counter(w.pos_.encode('ascii') for w in tokens)
        ctr['words_to'] = sum(1 for w in words if word == 'to')


        words_lemma = [w.lemma_.encode('ascii') for w in tokens]
        ctr['words_nostop'] = sum(1 for w in tokens if not w.is_stop)
        ctr['words_primaryvb'] = sum(1 for w in words_lemma \
                                        if w in {'be', 'have', 'do'})


        for key,val in pos_ctr.iteritems():
            ctr['words_%s' % key.lower()] = val   # change all tags to lowercase
                                                  # to match auto-lowercased
                                                  # Postgres column names


    # no. of singular pronouns
    singulars = {'i', 'me', 'my', 'mine'}
    ctr['pronouns_singular'] = sum(1 for word in words if word in singulars)


    # no. of plural pronouns
    plurals = {'we', 'us', 'our', 'ours'}
    ctr['pronouns_plural'] = sum(1 for word in words if word in plurals)


    # MongoDB doesn't like numpy array or set...
    return (ctr, list(words),
                 list(words_lemma),
                 list(misspellings),
                 list(gb_spellings),
                 list(us_spellings))



def is_US(word):
    """
    : type word: str
    : rtype: bool
    :
    : Check whether a word is in the American English dictionary.
    :
    """

    d = enchant.Dict('en_US')
    return d.check(word) or d.check(word.capitalize()) or d.check(word.upper())



def is_GB(word):
    """
    : type word: str
    : rtype: bool
    :
    : Check whether a word is in the British English dictionary.
    :
    """

    d = enchant.Dict('en_GB')
    return d.check(word) or d.check(word.capitalize()) or d.check(word.upper())



def get_topN(words, N):
    """
    :type words: List[str]
    :rtype: dict
    :
    : Gets the top N frequently used words.
    :
    """

    ctr = Counter(words)

    return dict(sorted(ctr.iteritems(), key=itemgetter(1), reverse=True)[:N])



def compute_freqs_features(results, bag_of_words, label):
    """
    :type results: List[json]
    :type bag_of_words: List[str]
    :rtype: pandas.DataFrame
    :
    : Count word occurrences.
    :
    """

    # should use a sparse matrix when there are many samples
    df = pd.DataFrame(columns = bag_of_words)

    for result in results:
        ctr = Counter(result['words_lemma'])
        dic = pd.Series(ctr.values(), index=ctr.keys(),
                        name=result['author'] + \
                             '_'.join(result['type']) + \
                             str(result['uid']))
        dic /= dic.sum()
        df = df.append(dic[bag_of_words])

    df['label'] = label

    df.fillna(0, inplace=True)

    return df



def compute_stylo_features(df_orig, label, chunk_size=500):
    df = df_orig.copy()

    # sentence features currently not using:
    # df['periods_onespace/sentences_punkt']  = df['periods_onespace']  / \
    #                                           df['sentences_punkt']
    # df['periods_twospaces/sentences_punkt'] = df['periods_twospaces'] / \
    #                                           df['sentences_punkt']
    # df['words/sentences_punkt'] = df['words'] / df['sentences_punkt']


    # sentense features currently using:
    df['periods_onespace/words']  = df['periods_onespace']  / \
                                    df['words'] * chunk_size
    df['periods_twospaces/words'] = df['periods_twospaces'] / \
                                    df['words'] * chunk_size
    df['sentences_punkt/words'] = df['sentences_punkt']     / \
                                  df['words'] * chunk_size


    # word features
    df['words_nostop/words']    = df['words_nostop']    / \
                                  df['words'] * chunk_size
    df['words_primaryvb/words'] = df['words_primaryvb'] / \
                                  df['words'] * chunk_size
    df['words_to/words']        = df['words_to']        / \
                                  df['words'] * chunk_size


    # pos features
    if 'words_noun' in df.columns:  # used Spacy .pos_
        word_types = {'verb': ['aux', 'verb'],
                      'noun': ['noun', 'pron', 'propn'],
                      'desc': ['adj', 'adp', 'adv', 'num'],
                      'det' : ['det'],
                      'conj': ['conj', 'intj', 'sconj', 'part']}

    else:  # Stanford POS tagger or Spacy .tag_
        word_types = {'verb': ['vb', 'vbd', 'vbn', 'vbp', 'vbz'],
                      'noun': ['nn', 'nns', 'nnp', 'nnps',
                               'prp', 'prp$', 'pp$'],  # noun, pronoun
                      'desc': ['jj', 'jjr', 'jjs', 'rb',
                               'rbr', 'rbs', 'wdt', 'wp',
                               'wp$', 'wrb'],   # adj, adv, 'which', 'how', etc
                      'det' : ['dt', 'pdt'],                                                         # determiner
                      'conj': ['cc', 'in']}                                                          # conjugate-like

    cnames  = set(df.columns)

    col_map = dict()
    for key, lst in word_types.iteritems():
        cname = 'words_%s' % key
        col_map[cname] = set('words_' + c for c in lst)
        df['%s/words'%cname] = df[list(cnames & col_map[cname])].sum(axis=1) / \
                               df['words'] * chunk_size


    # spelling feature
    df['words_gb/words'] = df['words_gb'] / df['words'] * chunk_size
    df['words_us/words'] = df['words_us'] / df['words'] * chunk_size


    # y label
    df['label'] = label

    df.index = df['author'] + \
               df['type'].apply(lambda lst: '_'.join(lst)) + \
               df['uid'].astype(str)

    df.drop(df_orig.columns, axis=1, inplace=True)

    return df




if __name__ == "__main__":

    pass
