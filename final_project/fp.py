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

    Doctests/Examples
    --------
    >>> query = "SELECT t1.ObjName FROM objects as t1, spectra as t2 WHERE (t1.ObjID = t2.ObjID) AND (t2.UT_Date > 19850101) AND (t2.UT_Date < 20180101) AND (t2.Min < 4500) and (t2.Max > 7000) AND (t2.Filename NOT LIKE '%gal%');"
    >>> result = mysql_query(dbp.usr, dbp.pswd, dbp.db, query)
    >>> len(result)
    6979
    >>> type(result)
    <class 'list'>
    >>> type(result[0])
    <class 'dict'>

    >>> query2 = "SELECT ObjName FROM objects LIMIT 10;"
    >>> result2 = mysql_query(dbp.usr, dbp.pswd, dbp.db, query2)
    >>> len(result2)
    10
    >>> result2[1]['ObjName']
    SN 1954A
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