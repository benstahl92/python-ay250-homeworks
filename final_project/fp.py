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

    Doctests
    --------
    >>> query = "SELECT t1.ObjName FROM objects as t1, spectra as t2 WHERE (t1.ObjID = t2.ObjID) AND (t2.UT_Date > 20090101) AND (t2.UT_Date < 20180101) AND (t2.Min < 4500) and (t2.Max > 7000) AND(t1.TypeReference != 'NULL') AND (t2.Filename NOT LIKE '%gal%') AND (t1.DiscDate > DATE('2008-01-01'));"
    >>> len(mysql_query(dbp.usr, dbp.pswd, dbp.db, query))
    1483
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()