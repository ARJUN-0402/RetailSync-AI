import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///database/retailsync.db")
tables = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"', engine)
print("Tables:", tables["name"].tolist())

for table in tables["name"].tolist():
    count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", engine).iloc[0]["cnt"]
    print(f"  {table}: {count} rows")
