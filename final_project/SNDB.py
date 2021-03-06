# sql interaction
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Spectra(Base):
    '''
    interface to spectra table in SNDB
    '''

    __tablename__ = 'spectra'
    SpecID = Column(Integer, primary_key=True)
    ObjID = Column(Integer)
    Filename = Column(String)
    Filepath = Column(String)
    SNID_Subtype = Column(String)
    Min = Column(String)
    Max = Column(String)
    SNR = Column(Float)

class Objects(Base):
    '''
    interface to objects table in SNDB
    '''

    __tablename__ = 'objects'
    ObjID = Column(Integer, primary_key=True)
    ObjName = Column(String)
    Redshift_Gal = Column(Float)

def get_session(usr, pswd, host, db):
    '''
    starts an sqlalchemy session and returns it

    Parameters
    ----------
    usr : database username
    pswd : database password
    host : database host
    db : database name

    Returns
    -------
    session : sqlalchemy session through which queries can be run against database
    '''

    engine = create_engine("mysql://{}:{}@{}/{}".format(usr, pswd, host, db))
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
