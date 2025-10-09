import os, sqlite3, psycopg2, psycopg2.extras, json

SQLITE = "db.sqlite3"
PGURL  = os.environ["DATABASE_URL"]

src = sqlite3.connect(SQLITE)
src.row_factory = sqlite3.Row
dst = psycopg2.connect(PGURL, sslmode="require")
dcur = dst.cursor()

def copy(table, cols):
    rows = src.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
    if not rows: return
    ph = ", ".join(["%s"]*len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT DO NOTHING"
    data = []
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if c == "meta" and isinstance(v, str):
                try: v = json.loads(v)
                except: pass
            vals.append(v)
        data.append(tuple(vals))
    dcur.executemany(sql, data)
    dst.commit()
    print(f"migrated {len(rows)} rows -> {table}")

# order matters for FKs
copy("users", ["id","name","mobile","password_hash","email","address","is_banned","is_verified","balance","locked_balance","verification_code","verify_expires","created_at"])
copy("refresh_tokens", ["jti","user_id","issued_at","expires_at","revoked"])
copy("wallet_txns", ["id","user_id","kind","amount","balance_after","locked_after","status","ref","note","meta","created_at"])
copy("withdrawal_requests", ["id","user_id","amount","net_amount","fee_amount","upi","payout_id","status","reason","created_at","updated_at"])
# fee_pool is single-row; we’ll just ensure it exists already in PG

src.close(); dcur.close(); dst.close()
print("✅ migration done")
