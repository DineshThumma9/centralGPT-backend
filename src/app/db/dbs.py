from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from src.app.models.schema import User
from sqlmodel import SQLModel

DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autoflush = False, autocommit = False,bind = engine)
Base = declarative_base()




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_table():
    SQLModel.metadata.create_all(bind=engine)


from sqlalchemy import text
from src.app.db.dbs import engine, create_table


def test_db_connection():
    try:
        # Test basic connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print(f"Database connection successful: {result.scalar() == 1}")

            # Check if PostgreSQL is running properly
            pg_version = connection.execute(text("SELECT version()")).scalar()
            print(f"PostgreSQL version: {pg_version}")

        # Create tables if they don't exist
        create_table()
        print("Tables created successfully")

    except Exception as e:
        print(f"Database connection error: {e}")


# Run the test
test_db_connection()

