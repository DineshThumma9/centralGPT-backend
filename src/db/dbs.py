import logging
import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

logger = logging.getLogger("database")
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yourpassword@localhost:5432/postgres")

try:
    logger.info(f"Creating database connection to {DATABASE_URL.split('@')[-1]}")
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 5
        }
    )

    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
    logger.info("Database connection established")

except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        logger.debug("Database session created")
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        logger.debug("Database session closed")
        db.close()


def create_all_tables():
    try:
        logger.info("Creating database tables")
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise
