# src/main.py

from src.app.application import app
from src.app.router import basic, auth
from src.app.db.dbs import create_table


create_table()
app.include_router(basic.router)
app.include_router(auth.router, prefix="/auth")

