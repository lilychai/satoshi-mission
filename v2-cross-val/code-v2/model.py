"""
:
: Author differentiation model.
:
"""


from __future__ import division   # no need to worry about integer division

from dbio import DBIO
import json
import warnings
import time

import spacy
from spacy.en import English     # takes a long time to initialise so save as an attribute

from helper_functions import bag_of_sentences
from helper_functions import strip_misc_periods
from helper_functions import create_chunks
from helper_functions import crunch_statistics
from helper_functions import get_topN, compute_freqs_features, compute_stylo_features

import pandas as pd
import numpy as np
from random import shuffle
from itertools import chain

from sklearn.svm import SVC, OneClassSVM, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, average_precision_score

import argparse


class MySVCModel(object):

    def __init__(self, num_class=2):
        """
        :type num_classes: int
        :rtype: None
        """

        self.__ctrl__ = None
        self.__case__ = None

        with open('../.dbname', 'r') as f:
            self.__DB_NAME__ = json.load(f)['dbname']

        self.__MG_DOCS_COLL__   = 'raw-docs'           # raw docs
        self.__MG_SENTS_COLL__  = 'bag-of-sents'       # raw sentences
        self.__MG_TOKENS_COLL__ = 'sample-tokens'      # clean tokens (words)
        self.__PG_STATS_TBL__   = 'stats'              # stylometric features
        self.__PG_RESULTS_TBL__ = 'results_' + \
                                  str(num_class) + \
                                  'class'              # cross val results
        self.__PG_PROBAS_TBL__  = 'probabilities'      # cross val probabilities


        self.__model__ = Pipeline([ \
                                 # ('scaler2', StandardScaler()),
                                 # ('scaler', MinMaxScaler()),
                                 # ('scaler3', Normalizer()),
                                  ('classifier', SVC(probability=True,
                                                     kernel='poly',
                                                     degree=2,
                                                     class_weight='balanced') \
                                                 if num_class-1 \
                                            else OneClassSVM(kernel='rbf',
                                                             nu=0.7,
                                                             gamma=1./250))
                                 ])

        print 'Instantiated classifier %s.' % \
              self.__model__.named_steps['classifier'].__class__.__name__


        self.__io__ = DBIO(MG_DB_NAME=self.__DB_NAME__,
                           PG_DB_NAME=self.__DB_NAME__)

        self.__tagger__ = None     # initialise if re-creating samples
        self.__bootstrap__ = None  # initialise in fit



    def fit(self, author1, author2, wts1=None, wts2=None,
                              bootstrap=False, verbose=False):
        """
        :type author1: str
        :type author2: str
        :type wts1: str/List[str]
        :type wts2: str/List[str]
        :type verbose:bool
        :rtype: bool
        :
        : Prepares databases and tables/collections.
        :
        """

        self.__bootstrap__ = bootstrap

        cases = []
        for i, (author, wts) in enumerate([(author1, wts1), (author2, wts2)]):
            if not wts:
                wts = [wt.encode('ascii') \
                       for wt in self.__io__.mg_distinct(self.__MG_DOCS_COLL__,
                                                         'type',
                                                         { 'author':author } )]

            if not isinstance(wts, list):
                wts = [wts]

            cases += (author, wts, (1,-1)[i]),   # use 1, -1 to match output
                                                 # from sklearn's OneClassSVM


        self.__ctrl__ = cases[0]    # assign label 1 in y vector
        self.__case__ = cases[1]    # assign be label 0 in y vector
        self.__MG_TOKENS_COLL__ += '-' + cases[0][0] + \
                                   '-' + cases[1][0] + \
                                   '-' + \
                                   ''.join(wt[:3] for wt in cases[0][1]) + \
                                   '-' + \
                                   ''.join(wt[:3] for wt in cases[1][1]) + \
                                   '-' + \
                                   ('nobs','bs')[bootstrap]

        self.__PG_STATS_TBL__   += '_' + cases[0][0] + \
                                   '_' + cases[1][0] + \
                                   '_' + \
                                   ''.join(wt[:3] for wt in cases[0][1]) + \
                                   '_' + \
                                   ''.join(wt[:3] for wt in cases[1][1]) + \
                                   '_' + \
                                   ('nobs','bs')[bootstrap]



        if verbose:
            print 'Control:', self.__ctrl__
            print 'Case:   ', self.__case__
            print 'Saving tokens to', self.__MG_TOKENS_COLL__
            print 'Saving stats to', self.__PG_STATS_TBL__

        return self.__prep_sents__(verbose=verbose) # err in preparing sentences




    def __prep_sents__(self, verbose=False):
        """
        :type verbose: bool
        :rtype: bool
        :
        : Prepares bags of raw sentences if not already done so.
        : (lump docs by the same auther and of the same type into the same bag)
        :
        """

        for author, wts, _ in (self.__ctrl__, self.__case__):

            for wt in wts:    # process type by type so sents can be reused

                if verbose:
                    print "Checking for %s's %s sentences..." % (author, wt)

                query_dic = { 'author' : author,
                              'type'   : wt      }


                if self.__io__.mg_find(self.__MG_SENTS_COLL__,
                                       query_dic).count()       == 0:

                    print "  No sentences found. Preparing from raw docs..."

                    query_results = self.__io__.mg_find(self.__MG_DOCS_COLL__,
                                                        query_dic)

                    # gather raw lines
                    contents = []
                    for result in query_results:
                        if isinstance(result['content'], list):
                            contents += result['content']  # list of lines
                        else:
                            contents += result['content'], # entire doc = a str

                    if not contents:
                        print '  Error: No docs found.'
                        return False


                    # create a bag of sentences from contents
                    sentences = bag_of_sentences(contents)

                    if not self.__io__.mg_insert_one(self.__MG_SENTS_COLL__,
                                                   query_dic,
                                                   dict({'sentences':sentences},
                                                         **query_dic),
                                                   verbose=True):

                        print 'Error in saving sentences.'
                        return False

        return True



    def cross_validate(self, chunk_size=500, k=3,
                             verbose=False, recreate_samples=False):
        """
        :type chunk_size: int
        :type k: int
        :type verbose: bool
        :type recreate_samples: bool
        :rtype: bool
        :
        : Perform k-fold cross validation.
        :
        """

        # create results table if not exist
        pg_header_dic = {'author1': 'varchar(20)',
                         'wts1'    : 'text[]',
                         'author2': 'varchar(20)',
                         'wts2'    : 'text[]',
                         'bs'     : 'boolean',
                         'fold'   : 'int'    }

        # create table to store stylometric results if not already exist
        if isinstance(self.__io__.pg_find("""
                                          SELECT COUNT(*) FROM %s
                                          """ % self.__PG_RESULTS_TBL__),
                                          type(None)):

            self.__io__.recreate_tables(PG_TBL_NAME   = self.__PG_RESULTS_TBL__,
                                        pg_header_dic = pg_header_dic,
                                        verbose=True)


        # create table to store prediction probabilities if not already exist
        if self.__model__.named_steps['classifier'].__class__.__name__ not in \
                                        set(['OneClassSVM', 'LinearSVC']) and \
           isinstance(self.__io__.pg_find("""
                                          SELECT COUNT(*) FROM %s
                                          """ % self.__PG_PROBAS_TBL__),
                                          type(None)):

            self.__io__.recreate_tables(PG_TBL_NAME   = self.__PG_PROBAS_TBL__,
                                        pg_header_dic = dict({'uid': 'int'},
                                                             **pg_header_dic),
                                        verbose=True)


        # create samples if not already exist or recreate samples when forced to
        if self.__io__.mg_find(self.__MG_TOKENS_COLL__, {}).count() == 0 or \
           recreate_samples:

            print 'Initialising Spacy POS tagger...'
            self.__tagger__ = English()     # spacy tagger


            # recreate tables (role = "trian", "test")
            self.__io__.recreate_tables(MG_COLL_NAME = self.__MG_TOKENS_COLL__,
                                        PG_TBL_NAME  = self.__PG_STATS_TBL__,
                                        pg_header_dic={'fold'  : 'int',
                                                       'role'  : 'varchar(5)',
                                                       'uid'   : 'int',
                                                       'author': 'varchar(20)',
                                                       'type'  : 'text[]'},
                                                       verbose=True)

            # split raw sentences into folds
            print 'Splitting folds...'
            folds_dic = self.__create_folds__(k, verbose=verbose)

            if verbose:
                for key, item in folds_dic.iteritems():
                    print '  %s: %i folds (%s sentences)' % \
                          (key, len(item), [len(lst) for lst in item])



            # create chunks then clean, tokenize and count words in each chunk
            print 'Creating and cleaning chunks...'
            if not self.__create_samples__(folds_dic, k, chunk_size,
                                           bootstrap=self.__bootstrap__,
                                           verbose=verbose):
                print 'Error in creating samples.'
                return False



        for fold in xrange(k):

            # create feature matrix
            print 'Preparing feature matrix.'
            dfs = self.__prepare_df__(fold, verbose=verbose)
            X_train = dfs[0].drop('label', axis=1)
            y_train = dfs[0]['label']
            X_test  = dfs[1].drop('label', axis=1)
            y_true  = dfs[1]['label']


            # train model
            print 'Fit and predict...'
            self.__model__.fit(X_train, y_train)
            y_pred = self.__model__.predict(X_test)


            error = False
            with warnings.catch_warnings(record=True) as w:

                # compute metrics
                CM        = confusion_matrix(y_true, y_pred)
                f1        = f1_score(y_true, y_pred)
                accuracy  = accuracy_score(y_true, y_pred)
                recall    = recall_score(y_true, y_pred)
                precision = average_precision_score(y_true, y_pred)

                if len(w):  # caught f1 UndefinedMetricsWarning from sklearn
                    print w[-1].message
                    error = True


            print '  f1 score       :', f1
            print '  accuracy score :', accuracy
            print '  recall score   :', recall
            print '  area under the precision-recall curve:', precision


            if self.__model__.named_steps['classifier'].__class__.__name__ \
            not in set(['OneClassSVM', 'LinearSVC']):

                probas = self.__model__.predict_proba(X_test)
                print '  probabilities:\n', probas


            # save results to postgres
            header_dic = {'author1' : self.__ctrl__[0],
                          'wts1'    : self.__ctrl__[1],
                          'author2' : self.__case__[0],
                          'wts2'    : self.__case__[1],
                          'bs'      : self.__bootstrap__,
                          'fold'    : fold             }

            # CM: row = true, col = pred, ind 0 = case, ind 1 = ctrl
            detail_dic = { 'tp': CM[1][1],    # ctrl classified as ctrl
                           'fp': CM[0][1],
                           'fn': CM[1][0],    # ctrl classified as case
                           'tn': CM[0][0],
                           'f1'      : f1,
                           'accuracy': accuracy,
                           'recall'  : recall,
                           'avg_prec': precision,
                           'und_err' : error      }


            if self.__model__.named_steps['classifier'].__class__.__name__ \
            not in set(['OneClassSVM', 'LinearSVC']):

                detail_dic.update({ 'pdiff_mean': np.mean(abs(probas[:,0] - \
                                                              probas[:,1])),

                                    'pdiff_std' : np.std(abs(probas[:,0] - \
                                                             probas[:,1])) })



            if not self.__io__.pg_insert_one(self.__PG_RESULTS_TBL__,
                                             header_dic,
                                             detail_dic,
                                             verbose=True):
                print 'Error in saving results.'
                return False


            # store probablilities if using two-class SVC
            if self.__model__.named_steps['classifier'].__class__.__name__ \
            not in set(['OneClassSVM', 'LinearSVC']):

                for i, row in enumerate(probas):
                    if not self.__io__.pg_insert_one(self.__PG_PROBAS_TBL__,
                                            dict({ 'uid' : i }, **header_dic),
                                            detail_dic = { 'y_pred': y_pred[i],
                                                           'case_0': row[0],
                                                           'ctrl_1': row[1] },
                                            verbose=True):

                        print 'Error in saving results.'
                        return False

        return True



    def __create_folds__(self, k=3, verbose=False):
        """
        :type k: int
        :type verbose: bool
        :rtype: dict
        :
        : Divide sentences into k folds.
        :
        """

        folds_dic, target_count = {}, 0

        for author, wts, _ in (self.__ctrl__, self.__case__):
            query_dic = { 'author' : author,
                          'type'   : { '$in': wts } }

            query_results = self.__io__.mg_find(self.__MG_SENTS_COLL__,
                                                query_dic)

            sentences = [result['sentences'] for result in query_results]
            sentences = list(chain(*sentences))

            if verbose:
                print '  (%s, %s): %i sentences' % (author, wts, len(sentences))

            # randomly shuffle sentences
            shuffle(sentences)   # in place shuffle
            folds_dic[(author, str(wts))] = [sentences[i::k] for i in xrange(k)]

        return folds_dic



    def __create_samples__(self, folds_dic, k=3, chunk_size=500,
                           bootstrap=False, verbose=False):
        """
        :type folds_dic: dict
        :type k: int
        :type chunk_size: int
        :type bootstrap: bool
        :type verbose: bool
        :rtype: bool
        :
        : Create a sample out of each chunk in each fold.
        :
        """

        # process fold by fold
        for fold in xrange(k):

            print '  Starting fold %i...' % fold

            # create training chunks
            chunks_train_dic, chunks_test_dic = {}, {}
            for author, wts, _ in (self.__ctrl__, self.__case__):

                chunks_train_dic[(author, str(wts))] = \
                  create_chunks(list(chain(*folds_dic[(author, str(wts))][1:])),
                                chunk_size,
                                0,
                                verbose=False)


            # compute amount to bootstrap
            target_count = max(len(chunks) \
                               for chunks in chunks_train_dic.values())


            for author, wts, _ in (self.__ctrl__, self.__case__):

                # bootstrap minority class to 50/50
                if bootstrap and \
                   len(chunks_train_dic[(author, str(wts))]) < target_count:

                    if verbose:
                        print '    Bootstrap minority to %i chunks.' % target_count

                    chunks_train_dic[(author, str(wts))] = \
                          create_chunks(list(chain(*folds_dic[(author,
                                                               str(wts))][1:])),
                                        chunk_size,
                                        target_count,
                                        verbose=False)


                # do not bootstrap test
                chunks_test_dic[(author, str(wts))] = \
                create_chunks(folds_dic[(author, str(wts))][0],
                              chunk_size,
                              0,
                              verbose=False)


                # prepare folds_dic for next fold-iteration
                folds_dic[(author, str(wts))] = \
                                        folds_dic[(author, str(wts))][1:] + \
                                        folds_dic[(author, str(wts))][:1]



            if verbose:
                for author, wts, _ in (self.__ctrl__, self.__case__):

                    print '    (%s, %s): %i train, %i test' % \
                          (author,
                           wts,
                           len(chunks_train_dic[(author, str(wts))]),
                           len(chunks_test_dic[(author, str(wts))]))




            # Clean, tokenize and compute stats for each chunk
            for author, wts, _ in (self.__ctrl__, self.__case__):

                for role, chunks \
                 in [('train', chunks_train_dic[(author, str(wts))]),
                     ('test',  chunks_test_dic[(author, str(wts))])]:

                    for i, chunk in enumerate(chunks,1):

                        print '    Processing %s %s %s chunk %i/%i...' % \
                                   (author, wts, role, i, len(chunks))


                        # clean chunks, tokenize and compute stats
                        stats_ctr, \
                        words, \
                        words_lemma, \
                        misspellings, \
                        gb_spellings, \
                        us_spellings = crunch_statistics(chunk, self.__tagger__)


                        query_dic = { 'fold'   : fold,
                                      'role'   : role,
                                      'uid'    : i,
                                      'author' : author,
                                      'type'   : wts }


                        entry_dic = { 'words': words,
                                      'words_lemma': words_lemma,
                                      'misspellings': misspellings,
                                      'us_spellings': us_spellings,
                                      'gb_spellings': gb_spellings   }


                        # save words in mongodb
                        if not self.__io__.mg_insert_one(self.__MG_TOKENS_COLL__,
                                                         query_dic,
                                                         dict(query_dic,
                                                              **entry_dic),
                                                         verbose=True):
                            print 'Error in saving words.'
                            return False


                        # save stats in postgres
                        if not self.__io__.pg_insert_one(self.__PG_STATS_TBL__,
                                                         query_dic,
                                                         stats_ctr,
                                                         verbose=True):
                            print 'Error in saving stats.'
                            return False

                        # break # break chunks

        return True



    def __prepare_df__(self, fold, verbose=False):
        """
        :type fold: int
        :type verbose: bool
        :rtype: pandas.DataFrame
        :
        : Prepare feature matrix.
        :
        """

        # compute bag-of-word
        query_results = self.__io__.mg_find( self.__MG_TOKENS_COLL__,
                                             {'fold'  : fold,
                                              'role'  : 'train',
                                              'author': self.__ctrl__[0],
                                              'type'  : self.__ctrl__[1]  } )

        words_lists = [result['words_lemma'] for result in query_results]
        words = list(chain(*words_lists))
        bag_of_words = get_topN(words, 250).keys()

        if verbose:
            print '  No. of words lists:', len(words_lists)
            print '  No. of lemmatized words:', len(words)
            print '  No. of unique lemmatized words', len(set(words))


        # prepare features dataframe for model
        dfs = []
        for role in ('train', 'test'):

            df_stylo, df_freq = pd.DataFrame(), pd.DataFrame()

            # get stylometric features
            for author, wts, label in (self.__ctrl__, self.__case__):


                # train one-class SVM with only literature from ctrl
                if \
                self.__model__.named_steps['classifier'].__class__.__name__ == \
                                                                 'OneClassSVM' \
                and role == 'train' \
                and label == self.__case__[2]:

                    continue


                # get term frequency features
                q = '''
                    SELECT * FROM %s
                    WHERE fold   = %i
                      AND role   = '%s'
                      AND author = '%s'
                      AND type   = ARRAY%s;
                    ''' % (self.__PG_STATS_TBL__, fold, role, author, wts)

                df = self.__io__.pg_find(q)
                if isinstance(df, type(None)):
                    print 'Error: no stats found.'
                    return False

                df_stylo = df_stylo.append(compute_stylo_features(df, label))


                # get term frequency features
                results = [results for results in \
                           self.__io__.mg_find( self.__MG_TOKENS_COLL__,
                                                { 'fold'  : fold,
                                                  'role'  : role,
                                                  'author': author,
                                                  'type'  : wts } ) ]

                df_freq = df_freq.append(compute_freqs_features(results,
                                                                bag_of_words,
                                                                label))


            # horizontally stack DataFrames
            dfs += df_stylo.join(df_freq.drop('label', axis=1)),


        if verbose:
            print '  df_train shape:', dfs[0].shape
            print '  df_test shape:', dfs[1].shape

        return dfs




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='My SVC Model')

    parser.add_argument('author1', metavar='A1', type=str,
                        help='Author 1 (ctrl).')

    parser.add_argument('author2', metavar='A2', type=str,
                        help='Author 2 (case).')

    parser.add_argument('-wts1', metavar='WTS1', dest='wts1', type=str,
                        default=None,
                        help='Writing types (default: all)', nargs='+')

    parser.add_argument('-wts2', metavar='WTS2', dest='wts2', type=str,
                        default=None,
                        help='Writing types (default: all)', nargs='+')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--bs', dest='bootstrap',
                                action='store_true',
                                help='Bootstrap to 50/50.')
    feature_parser.add_argument('--nobs', dest='bootstrap',
                                action='store_false',
                                help='No bootstrapping (default).')
    parser.set_defaults(bootstrap=False)


    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--1', dest='num_class',
                                action='store_false',
                                help='One-class SVM')
    feature_parser.add_argument('--2', dest='num_class',
                                action='store_true',
                                help='Two-class classifier (default).')
    parser.set_defaults(num_class=True)


    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--rs', dest='recreate_samples',
                                action='store_true',
                                help='Recreate samples.')
    feature_parser.add_argument('--nors', dest='recreate_samples',
                                action='store_false',
                                help=\
                                'Use existing samples if available (default).')
    parser.set_defaults(recreate_samples=False)



    args = parser.parse_args()


    model = MySVCModel(num_class=int(args.num_class)+1)


    # if no error in fit, perform cross-val
    if model.fit(args.author1, args.author2, args.wts1, args.wts2,
                 bootstrap=args.bootstrap, verbose=True):

        model.cross_validate(verbose=True,
                             recreate_samples=args.recreate_samples)
