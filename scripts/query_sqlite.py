import pandas as pd
import sqlite3

# Define the database and table names (must match the first script)
db_filename = 'prices.db'
table_name = 'house_prices'

# 1. Connect to the existing SQLite database
conn = sqlite3.connect(db_filename)

# 2. Use pandas to run a SQL query and load the results directly into a DataFrame
# This reads the entire table into 'df_loaded'
df_loaded = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# 3. Always close the database connection when done
conn.close()

# 4. Use the DataFrame as normal
print("Data successfully loaded from the SQLite database into a DataFrame!")
print(f"Shape of loaded DataFrame: {df_loaded.shape}")
print("\nFirst 5 rows:")
print(df_loaded.head())