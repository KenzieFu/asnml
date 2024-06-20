import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv('GOOGLE_CLOUD_SQL_USER','root') 
PASS = os.getenv('GOOGLE_CLOUD_PASS','') 
HOST = os.getenv('GOOGLE_CLOUD_HOST','localhost')  
DBNAME = os.getenv('GOOGLE_CLOUD_DATABASE','backend-asn') 

# URL_DATABASE = 'mysql+pymysql://asn:3Ha&*|RCQm|~Pp/X@34.101.41.205/backend-asn'
URL_DATABASE = f'mysql+pymysql://{USER}:{PASS}@{HOST}:3306/{DBNAME}'
engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

Base = declarative_base()  