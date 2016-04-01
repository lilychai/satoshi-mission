"""
:
: Functions for extracting text from pdf document.
:
"""


from __future__ import division        # no need to worry about integer division

from dbio import DBIO
import textract

import string, re
from datetime import datetime

import argparse


def extract_pdf(filename, author, date,
                writing_type='paper', recreateTables=False):
    """
    :type filename: str
    :type author: str
    :type date: str
    :type writing_type: str
    :type recreateTables: bool
    :rtype: bool
    :
    : Extract text from a pdf file.
    :
    """

    print 'Extracting', filename

    doc = textract.process(filename, method='pdftotext', layout=True,
                           encoding='ascii')   # output encoding;
                                               # input encoding is inferred


    # save raw text to MongoDB
    with open('../../.dbname', 'r') as f:
        DB_NAME      = json.load(f)['dbname']
        
    MG_COLL_NAME = 'raw-docs'

    io = DBIO(MG_DB_NAME=DB_NAME)
    if recreateTables:
        io.recreate_tables(MG_COLL_NAME = MG_COLL_NAME)

    query_dic = { 'author' : author,
                  'date'   : date,
                  'type'   : writing_type }

    return io.mg_insert_one(MG_COLL_NAME,
                            query_dic=query_dic,
                            insertion_dic=dict({'content': doc}, **query_dic),
                            verbose=True)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= \
            'Prepare all writings of the same type into a list of sentences.')

    parser.add_argument('filename', metavar='F', type=str,
                        help='File name.')

    parser.add_argument('author', metavar='A', type=str,
                        help='Author name.')

    parser.add_argument('date', metavar='DT', type=str,
                        help='Publication date (YYYY-MM-DD HH:mm:ss tz).')

    parser.add_argument('-wt', metavar='WT', dest='writing_type', type=str,
                        help='Writing type (default = paper).')

    parser.add_argument('-rt', metavar='RT', dest='recreateTables', type=bool,
                        default=False,
                        help='Recreate tables (default = False).')

    args = parser.parse_args()

    try:
        datetime.strptime('1970-01-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')

    except:
        print 'Error: incorrect date string.'
        raise

    extract_pdf(args.filename,
                author=args.author,
                date=args.date,
                writing_type=args.writing_type,
                recreateTables=args.recreateTables)
