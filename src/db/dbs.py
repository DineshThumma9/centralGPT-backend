from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger("database")

# Load environment variables
load_dotenv()

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yourpassword@localhost:5432/postgres")

# Create engine with proper timeout settings
try:
    logger.info(f"Creating database connection to {DATABASE_URL.split('@')[-1]}")
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Check connection before using from pool
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={
            "connect_timeout": 5  # Timeout after 5 seconds if can't connect
        }
    )


    # Add connection debugging
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault('query_start_time', []).append(
            logging.getLogger("database").debug(f"Executing SQL: {statement}"))


    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        logging.getLogger("database").debug(f"Finished executing SQL: {statement}")


    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
    logger.info("Database connection established")

except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    # Create a dummy engine for SQLModel.metadata.create_all to work in dev without DB
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


def get_db():
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


# Function to create all tables
def create_all_tables():
    try:
        logger.info("Creating database tables")
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise