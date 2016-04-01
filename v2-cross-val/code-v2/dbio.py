"""
:
: DBIO: an object that communicates with MongoDB and Postgres databases.
:
"""


from pymongo import MongoClient
import psycopg2 as pg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, AsIs

from itertools import chain, izip

from pandas import read_sql
import numpy as np
import json


class DBIO(object):

    def __init__(self, MG_DB_NAME=None, PG_DB_NAME=None):
        """
        :type MG_DB_NAME: str        -- MongoDB name
        :type PG_DB_NAME: str        -- Postgres DB name
        :rtype: None
        :
        : Constructor. Creates a Postgres database if given a non-existent
        : database name.
        :
        """

        self.__mg_db_name__ = MG_DB_NAME
        self.__pg_db_name__ = PG_DB_NAME


        self.__mg_conn__ = MongoClient()[MG_DB_NAME] if MG_DB_NAME else None

        try:
            if PG_DB_NAME:
                with open('../../../auth/postgres/postgres.json', 'r') as f:
                    PGCONN = json.load(f)
                    PGCONN['dbname'] = PG_DB_NAME

                self.__pg_conn__ = pg2.connect(**PGCONN)

            else:
                self.__pg_conn__ = None

        except:
            self.create_database(PG_DB_NAME=PG_DB_NAME)



    def __del__(self):
        """
        :rtype: None
        """

        del self.__mg_db_name__
        del self.__pg_db_name__

        del self.__mg_conn__
        del self.__pg_conn__



    def __str__(self):
        """
        :rtype str
        """

        return '[' + str(self.__mg_db_name__) + \
              ', ' + str(self.__pg_db_name__) + ']'



    def create_database(self, MG_DB_NAME=None, PG_DB_NAME=None, verbose=False):
        """
        :type MG_DB_NAME: str
        :type PG_DB_NAME: str
        :type verbose: bool
        :rtype: bool
        :
        : Create databases if not already exist.
        :
        """

        r = False

        # MongoDB:
        if MG_DB_NAME:
            self.__mg_db_name__ = MG_DB_NAME
            self.__mg_conn__ = MongoClient()[MG_DB_NAME]

            if verbose:
                print "Connected to existing or new MongoDB database '%s'" % \
                      MG_DB_NAME

            r = True


        # Postgres:
        if PG_DB_NAME:

            try:
                with open('../../../auth/postgres/postgres.json', 'r') as f:
                    PGCONN = json.load(f)
                    PGCONN['dbname'] = 'postgres'

                conn = pg2.connect(**PGCONN)
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                cur = conn.cursor()

                cur.execute("""
                            SELECT COUNT(*)
                            FROM pg_catalog.pg_database
                            WHERE datname = '%s';
                            """ % PG_DB_NAME)


                if cur.fetchall()[0][0] > 0:  # if sth other than [(0L,)]
                    if verbose:
                        print "Connecting to existing database '%s'." % \
                              PG_DB_NAME
                else:
                    cur.execute("CREATE DATABASE %s" % PG_DB_NAME)

                    if verbose:
                        print "Created Postgres database '%s'" % PG_DB_NAME

                cur.close()
                conn.close()
                r = True


            except Exception as e:

                if verbose:
                    print e.__doc__
                    print e.message

                r = False

            finally:

                if 'cur' in locals():  cur.close()
                if 'conn' in locals(): conn.close()


            with open('../../../auth/postgres/postgres.json', 'r') as f:
                PGCONN = json.load(f)
                PGCONN['dbname'] = PG_DB_NAME

            self.__pg_db_name__ = PG_DB_NAME
            self.__pg_conn__ = pg2.connect(**PGCONN)

        return r


    def recreate_tables(self, MG_COLL_NAME=None, PG_TBL_NAME=None,
                              pg_header_dic=None, verbose=False):
        """
        :type MG_COLL_NAME: str
        :type PG_TBL_NAME: str
        :type pg_header_dic: dict
        :type verbose: bool
        :rtype: bool
        :
        : Drop table/collection if exist and create a new table/collection.
        :
        """

        r = True

        # MongoDB: drop collection; no need to recreate -- instantiates on first access.
        if MG_COLL_NAME:

            try:

                self.__mg_conn__[MG_COLL_NAME].drop()
                if verbose:
                    print "Created new collection '%s'" % MG_COLL_NAME

                r = True

            except Exception as e:

                if verbose:
                    print e.__doc__
                    print e.message

                r = False


        # Postgres: drop and recreate table
        if PG_TBL_NAME:

            if not pg_header_dic:
                if verbose:
                    "Need to provide unique keys!"
                return r & False

            try:
                cur = self.__pg_conn__.cursor()

                PG_TBL_NAME = (PG_TBL_NAME,)              # so that can added to tuples

                HEADER_FIELDS, HEADER_TYPES = zip(*sorted(pg_header_dic.iteritems()))


                q = ('''
                     DROP TABLE IF EXISTS %s;

                     CREATE TABLE %s (

                     ''' + ', '.join(['%s %s'] * len(HEADER_FIELDS)) + \
                     '''
                     , primary key (

                     ''' + ', '.join(['%s'] * len(HEADER_FIELDS)) + \
                     '''
                     )
                     );
                     ''') % (PG_TBL_NAME + PG_TBL_NAME + \
                             tuple(chain(*sorted(pg_header_dic.iteritems())))+ \
                             HEADER_FIELDS)


                cur.execute(q)
                self.__pg_conn__.commit()

                if verbose:
                    "Created new table '%s'." % PG_TBL_NAME

                r &= True

            except Exception as e:

                if verbose:
                    print e.__doc__
                    print e.message

                r &= False

            finally:

                if 'cur' in locals():  cur.close()
                if 'conn' in locals(): conn.close()

        return r


    def mg_find(self, COLL_NAME, query_dic, verbose=False):
        """
        :type COLL_NAME: str
        :type query_dic: dict
        :type verbose: bool
        :rtype: cursor/json/None
        :
        : Execute MongoDB find query.
        :
        """

        if not self.__mg_db_name__:
            if verbose:
                'Database not specified.'
            return None

        try:
            tbl = self.__mg_conn__[COLL_NAME]
            return tbl.find( query_dic )

        except Exception as e:

            if verbose:
                print e.__doc__
                print e.message

            return None


    def mg_distinct(self, COLL_NAME, field, query_dic, verbose=False):
        """
        :type COLL_NAME: str
        :type field: str
        :type query_dic: dict
        :type verbose: bool
        :rtype: cursor/json/bool
        :
        : Execute MongoDB distinct query.
        :
        """

        if not self.__mg_db_name__:
            if verbose:
                'Database not specified.'
            return False

        try:
            tbl = self.__mg_conn__[COLL_NAME]
            return tbl.distinct( field, query_dic )

        except Exception as e:

            if verbose:
                print e.__doc__
                print e.message

            return False



    def mg_insert_one(self, COLL_NAME, query_dic, insertion_dic, verbose=False):
        """
        :type COLL_NAME: str
        :type query_dic: dict
        :type insertion_dic: dict
        :type verbose: bool
        :rtype: bool
        :
        : Insert one entry into a MongoDB collection,
        : or update entry if entry already exists.
        :
        """

        if not self.__mg_db_name__:
            if verbose:
                'Database not specified.'
            return False

        try:
            tbl = self.__mg_conn__[COLL_NAME]
            tbl.delete_one( query_dic )
            tbl.insert_one( insertion_dic )

            r = True

        except Exception as e:

            if verbose:
                print e.__doc__
                print e.message

            r = False


        return r    # common return, in case I want to add something


    def pg_find(self, query_str, verbose=False):
        """
        :type TBL_NAME: str
        :type query_str: dict
        :type verbose: bool
        :rtype: pandas.DataFrame or None
        :
        : Execute Postgres SELECT query.
        :
        """

        if not self.__pg_db_name__:
            if verbose:
                'Database not specified.'
            return None

        try:

            return read_sql(query_str, self.__pg_conn__)


        except Exception as e:

            if verbose:
                print e.__doc__
                print e.message

            return None



    def pg_insert_one(self, TBL_NAME, header_dic, detail_dic, verbose=False):
        """
        :type TBL_NAME: str
        :type header_dic: dict
        :type detail_dic: dict
        :type verbose: bool
        :rtype: bool
        :
        : Alter table if column does not exist.
        : Insert one entry into a Postgres table,
        : or update entry if entry already exists.
        :
        """

        if not self.__pg_db_name__:
            if verbose:
                'Database not specified.'
            return False


        NUM_HEADERS   = len(header_dic)
        NUM_DETAILS   = len(detail_dic)
        NUM_FIELDS    = NUM_HEADERS + NUM_DETAILS
        HEADER_FIELDS = tuple(sorted(header_dic.keys()))
        DETAIL_FIELDS = tuple(sorted(detail_dic.keys()))


        try:

            cur = self.__pg_conn__.cursor()

            # ADD COLUMNS in not exist
            q = '''
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '%s';
                ''' % TBL_NAME

            cur.execute(q)

            MISSING_FIELDS = tuple(set(DETAIL_FIELDS) - set(chain(*cur.fetchall())))

            types    = {type(1)              : 'int',
                        type(np.int64(1))    : 'int',
                        type(np.int32(1))    : 'int',
                        type(1.)             : 'numeric',
                        type(np.float64(1.)) : 'numeric',
                        type(np.float32(1.)) : 'numeric',
                        type('A')            : 'varchar(30)',  # 30 for now
                        type(True)           : 'boolean'}
            defaults = {type(1)              : '0',
                        type(np.int64(1))    : '0',
                        type(np.int32(1))    : '0',
                        type(1.)             : '0',
                        type(np.float64(1.)) : '0',
                        type(np.float32(1.)) : '0',
                        type('A')            : '' ,
                        type(True)           : 'false'}


            if MISSING_FIELDS:

                q = 'ALTER TABLE ' + TBL_NAME + \
                    ','.join([' ADD COLUMN %s %s DEFAULT %s' % t \
                              for t in [(f, types[type(detail_dic[f])],
                                            defaults[type(detail_dic[f])]) \
                                        for f in MISSING_FIELDS]]) + \
                    ';'

                cur.execute(q)


            # UPDATE if exist; INSERT if not
            q = ('''
                 INSERT INTO %s (

                 ''' + ', '.join(['%s'] * NUM_FIELDS) + \
                 '''
                 )

                 VALUES (

                 ''' + ', '.join(['%s(%s)s'] * NUM_FIELDS) + \
                 '''
                 )
                 ON CONFLICT ON CONSTRAINT %s_pkey DO
                 UPDATE SET (

                 ''' + ', '.join(['%s'] * NUM_DETAILS) + \
                 '''
                 ) = (

                 ''' + ', '.join(['%s(%s)s'] * NUM_DETAILS) + \
                 '''
                 );
                 ''') % ((TBL_NAME,) + \
                         (HEADER_FIELDS + DETAIL_FIELDS) + \
                         tuple(chain(*izip(['%'] * NUM_FIELDS,
                                          (HEADER_FIELDS + DETAIL_FIELDS)))) + \
                         (TBL_NAME,) + \
                         DETAIL_FIELDS + \
                         tuple(chain(*izip(['%'] * NUM_DETAILS,
                                           DETAIL_FIELDS       ))))


            dic = dict(header_dic, **detail_dic)

            cur.execute(q, dic)
            self.__pg_conn__.commit()

            r = True


        except Exception as e:

            if verbose:
                print e.__doc__
                print e.message

            r = False


        finally:

            if 'cur' in locals():  cur.close()


        return r



if __name__ == "__main__":

    io = DBIO()
    print 'Create databases/table/collection:'
    io.create_database(MG_DB_NAME='mongo_test',
                       PG_DB_NAME='pg_test', verbose=True)
    io.create_database(MG_DB_NAME='mongo_test',
                       PG_DB_NAME='pg_test', verbose=True)
    io.recreate_tables(MG_COLL_NAME='coll_test',
                       PG_TBL_NAME='tbl_test',
                       pg_header_dic={'a': 'int'}, verbose=True)

    print '\nTest insert:'
    io.mg_insert_one('coll_test', {'a': 1}, {'a': 1, 'b': 2}, verbose=True)
    io.pg_insert_one('tbl_test', {'a': 1}, {'b': 2 }, verbose=True)

    print '\nTest find:'
    print io.mg_find('coll_test', {'a': 1}, verbose=True).count()
    print io.pg_find('SELECT count(*) FROM tbl_test WHERE b = 2', verbose=True)

    print '\nTest replace:'
    io2 = DBIO(MG_DB_NAME='mongo_test', PG_DB_NAME='pg_test')
    io2.mg_insert_one('coll_test', {'a': 1}, {'a': 1, 'b': 20}, verbose=True)
    io2.pg_insert_one('tbl_test', {'a': 1}, {'b': 20 }, verbose=True)

    print '\nTest re-create table:'
    io2.recreate_tables(MG_COLL_NAME='coll_test',
                        PG_TBL_NAME='tbl_test',
                        pg_header_dic={'a':'int'}, verbose=True)

    ## clean up:
    ## mongo shell: use mongo_test; db.dropDatabase();
    ## psql shell:  drop database pg_test;
