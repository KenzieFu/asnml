from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# URL_DATABASE = 'mysql+pymysql://asn:3Ha&*|RCQm|~Pp/X@34.101.41.205/backend-asn'
URL_DATABASE = 'mysql+pymysql://asn:3Ha&*|RCQm|~Pp/X@34.101.41.205:3306/backend-asn'
engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

Base = declarative_base()  