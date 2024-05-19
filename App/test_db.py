# test_db.py


import sqlite3

conn = sqlite3.connect('./test.db')
cursor = conn.cursor()
cursor.execute("SELECT username, password FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
