import pandas as pd
import sqlite3

# Create a sample CSV file for demonstration if none is provided
csv_filename = 'price_clean.csv'

# Code to convert CSV to SQLite DB
db_filename = 'prices.db'
table_name = 'house_prices'

# 1. Read the CSV
df = pd.read_csv(csv_filename)

# 2. Create/Connect to the SQLite database
conn = sqlite3.connect(db_filename)

# 3. Write the dataframe to the database
df.to_sql(table_name, conn, if_exists='replace', index=False)

# 4. Verify the contents
query_result = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

print(f"Database '{db_filename}' created with table '{table_name}'.")
print("\nContents of the table:")
print(query_result)