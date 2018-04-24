# standard imports
import pymysql as sql

# login credentials of MySQL database
import db_params as dbp



def mysql_query(usr, pswd, db, query):
    '''
    execute a mysql query on a given database and return the result as a list containing all retrieved rows as dictionaries

    Parameters
    ----------
    usr : user name for database access (typically accessed from a param file, see example)
    pswd : password for given user name (typically accessed from a param file, see example)
    db : database name (typically accessed from a param file, see example)
    query : valid MySQL query

    Returns
    -------
    results : list of all retrieved results (each as a dictionary)
    '''

    # connect to db
    connection = sql.connect(user = usr, password = pswd, db = db, cursorclass = sql.cursors.DictCursor)

    # issue query and get results
    with connection.cursor() as cursor:
        cursor.execute(query)
        results = cursor

    # return results
    return list(results)

def main(query):
    '''
    Parameters
    ----------
    query : valid MySQL query

    Returns
    -------

    '''

    # execute query against database and retrieve results
    print('\nExecuting query...')
    results = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
    print('{} results retrieved'.format(len(results)))