"""
:
: Script for scraping and extracting Satoshi's e-mails and forum posts.
:
"""


from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup, NavigableString
import HTMLParser

from dbio import DBIO

from collections import Counter


# ##--- Fetch URLs ---
ROOT_URL = 'http://satoshi.nakamotoinstitute.org'

# fetch urls for emails
req = requests.get(ROOT_URL + '/emails/cryptography/')
soup = BeautifulSoup(req.content, 'html.parser')
tags = soup.findAll('ol')[0].findAll('a')
urls1 = [tag['href'] for tag in tags]

# fetch urls for forum posts
req = requests.get(ROOT_URL + '/posts/')
soup = BeautifulSoup(req.content, 'html.parser')
tags = soup.select('ul')[1].findAll('a')
urls2 = [tag['href'] for tag in tags]

# sanity check:
print 'No. of e-mails \t', len(urls1)
print 'No. of posts   \t', len(urls2)
print '-------------------'
print 'Total          \t', len(urls1) + len(urls2)
print '\n'
# print urls1[-1]


##--- Grab everything first and store in MongoDB ---

# access/instantiate database & collection

author  = 'satoshi'
DB_NAME = 'satoshi3'
MG_COLL_NAME = 'web-scrape'

io = DBIO()
io.create_database(MG_DB_NAME=DB_NAME, verbose=True)
io.recreate_tables(MG_COLL_NAME=MG_COLL_NAME)

for writing_type, source in (('email', urls1), ('forum', urls2)):
    for i,url in enumerate(source,1):
        print '  Downloading %s %i/%i' % (writing_type, i, len(source))

        io.mg_insert_one( MG_COLL_NAME,
                          { 'author': '' },        # dummy query
                          { 'author': author,
                            'type': writing_type,
                            'content': requests.get(ROOT_URL + url).content },
                          verbose=True)


# sanity check:
print 'No. of e-mails \t', io.mg_find(MG_COLL_NAME,
                                      { 'author': 'satoshi',
                                        'type'  : { '$eq': 'email' } } ).count()
print 'No. of posts   \t', io.mg_find(MG_COLL_NAME,
                                      { 'author': 'satoshi',
                                        'type'  : { '$eq': 'forum' } } ).count()
print '\n'



##--- Parse HTML for content ---

MG_SRC_COLL  = MG_COLL_NAME
MG_DEST_COLL = 'raw-docs'

html_parser = HTMLParser.HTMLParser()


# parse emails
writing_type = 'email'

print 'Parsing %s...' % writing_type

lst_len = []                           # for sanity check
for item in io.mg_find( MG_SRC_COLL, { 'author': author,
                                       'type'  : { '$eq': writing_type } } ):

    soup = BeautifulSoup(item['content'], 'html.parser')

    # will split by <br \>
    lst = list(desc.string for desc in \
               soup.select('.email-text-monospace')[0].descendants)

    date = soup.select('#CONTENT h3 em')[0].string

    # remove None and leading quotes
    cleanlst = [html_parser.unescape(line.encode('ascii', 'ignore')) \
                for line in lst \
                 if line \
                and not line.endswith('wrote:') \
                and not line.startswith('>')]


    # remove trailing signature and insert into clean table
    i = ([len(cleanlst)] + [i for i,line in enumerate(cleanlst) \
                               if line.startswith('-----')])[-1]
                                                         # in case no signature
    query_dic = { 'author': author,
                  'type': writing_type,
                  'date': date }

    io.mg_insert_one(MG_DEST_COLL,
                     query_dic=query_dic,
                     insertion_dic=dict({'content': cleanlst[:i]},**query_dic),
                     verbose=True)

    lst_len += len(cleanlst),


# sanity check:
print 'Total no. of docs:', len(lst_len)
print 'Doc lengths:', lst_len         # verify that each email is different
print '\n'



# parse forum posts

writing_type = 'forum'

print 'Parsing %s...' % writing_type

ctr = Counter()                                # for sanity check
for item in io.mg_find(MG_COLL_NAME, { 'author': author,
                                       'type'  : { '$eq': writing_type } } ):

    soup = BeautifulSoup(item['content'], 'html.parser')

    tag = soup.select('#CONTENT div')[0]


    # ignore reposts -- satoshi reposting other people's question
    # (i.e. not satoshi's writing)
    if tag.find('h3').string.startswith('Repost:'):
        ctr['repost'] += 1
        continue

    date = tag.select('h5 em')[0].string
    source = tag.find('h4').string
    ctr[source] += 1


    # p2p foundation posts
    if source == u'P2P Foundation':
        cleanlst = [html_parser.unescape(desc.string.encode('ascii', 'ignore'))\
                    for desc in tag.descendants \
                     if isinstance(desc, NavigableString)]
        cleanlst = cleanlst[cleanlst.index('Original Post')+1:]


    # bitcointalk posts
    else:
        cleanlst = [html_parser.unescape(item.string.encode('ascii', 'ignore'))\
                    for item in tag.children \
                     if isinstance(item, NavigableString)]
        cleanlst = [s for s in cleanlst if not s.startswith('>')]


    # save to database
    query_dic = { 'author': author,
                  'type': writing_type,
                  'date': date }

    io.mg_insert_one(MG_DEST_COLL,
                     query_dic=query_dic,
                     insertion_dic=dict({'content': cleanlst},**query_dic),
                     verbose=True)

print 'Doc count:', ctr
