# admin.py
from flask import Blueprint, request, jsonify
from functools import wraps
import os
from database import run_query
from writer import enqueue_write

bp = Blueprint("admin", __name__, url_prefix="/admin")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")  # set a real secret in .env!

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-Admin-Token")
        if token != ADMIN_TOKEN:
            return jsonify({"error": "admin auth failed"}), 403
        return fn(*args, **kwargs)
    return wrapper

def _audit(action, actor, target, details):
    ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "-"
    enqueue_write(
        "INSERT INTO audit_log (action, actor, target, details, ip) VALUES (?,?,?,?,?)",
        (action, actor, target, details, ip)
    )

# ---------- Users ----------
@bp.get("/users")
@admin_required
def list_users():
    q = (request.args.get("q") or "").strip()
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    params = []
    sql = "SELECT id, name, mobile, aadhar, pan, balance, is_banned FROM users"
    if q:
        sql += " WHERE name LIKE ? OR mobile LIKE ? OR pan LIKE ?"
        like = f"%{q}%"
        params.extend([like, like, like])
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = run_query(sql, tuple(params), fetch=True)
    users = [
        {"id": r[0], "name": r[1], "mobile": r[2], "aadhar": r[3], "pan": r[4], "balance": r[5], "is_banned": bool(r[6])}
        for r in rows
    ]
    return jsonify({"users": users, "limit": limit, "offset": offset})

@bp.post("/user/ban")
@admin_required
def user_ban():
    data = request.json or {}
    mobile = data.get("mobile")
    reason = data.get("reason", "")
    if not mobile:
        return jsonify({"error": "mobile required"}), 400
    ok, _, wait_log, eta, err = enqueue_write(
        "UPDATE users SET is_banned=1 WHERE mobile=?", (mobile,)
    )
    _audit("user_ban", "admin", mobile, reason)
    if not ok:
        return jsonify({"error": err or "ban failed", "wait_log": wait_log, "eta": eta}), 400
    return jsonify({"message": f"user {mobile} banned ✅", "wait_log": wait_log, "eta": eta})

@bp.post("/user/unban")
@admin_required
def user_unban():
    data = request.json or {}
    mobile = data.get("mobile")
    if not mobile:
        return jsonify({"error": "mobile required"}), 400
    ok, _, wait_log, eta, err = enqueue_write(
        "UPDATE users SET is_banned=0 WHERE mobile=?", (mobile,)
    )
    _audit("user_unban", "admin", mobile, "")
    if not ok:
        return jsonify({"error": err or "unban failed", "wait_log": wait_log, "eta": eta}), 400
    return jsonify({"message": f"user {mobile} unbanned ✅", "wait_log": wait_log, "eta": eta})

@bp.post("/user/delete")
@admin_required
def user_delete():
    data = request.json or {}
    mobile = data.get("mobile")
    pan = data.get("pan")
    if not (mobile or pan):
        return jsonify({"error": "mobile or pan required"}), 400
    if mobile:
        where, val = "mobile=?", mobile
    else:
        where, val = "pan=?", pan
    ok, _, wait_log, eta, err = enqueue_write(f"DELETE FROM users WHERE {where}", (val,))
    _audit("user_delete", "admin", val, "")
    if not ok:
        return jsonify({"error": err or "delete failed", "wait_log": wait_log, "eta": eta}), 400
    return jsonify({"message": f"user {val} deleted ✅", "wait_log": wait_log, "eta": eta})

@bp.post("/balance/adjust")
@admin_required
def balance_adjust():
    data = request.json or {}
    mobile = data.get("mobile")
    delta = float(data.get("delta", 0))
    note = data.get("note", "")
    if not mobile or delta == 0:
        return jsonify({"error": "mobile and non-zero delta required"}), 400

    # Update balance and insert transaction atomically (best-effort via two queued writes)
    ok1, _, w1, eta1, e1 = enqueue_write(
        "UPDATE users SET balance = balance + ? WHERE mobile=?", (delta, mobile)
    )
    ok2, _, w2, eta2, e2 = enqueue_write(
        "INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)",
        (mobile, "admin_adjust", abs(delta), "completed", note)
    )
    _audit("balance_adjust", "admin", mobile, f"delta={delta}; note={note}")
    ok = ok1 and ok2
    if not ok:
        return jsonify({"error": e1 or e2 or "adjust failed", "wait_log": w1 + w2, "eta": max(eta1, eta2)}), 400
    return jsonify({"message": f"balance adjusted by ₹{delta} for {mobile} ✅", "wait_log": w1 + w2, "eta": max(eta1, eta2)})

# ---------- Transactions ----------
@bp.get("/transactions")
@admin_required
def tx_list():
    mobile = request.args.get("mobile")
    status = request.args.get("status")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    sql = "SELECT id, mobile, type, amount, status, admin_note, timestamp FROM transactions"
    clauses, params = [], []
    if mobile:
        clauses.append("mobile=?")
        params.append(mobile)
    if status:
        clauses.append("status=?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = run_query(sql, tuple(params), fetch=True)
    data = [{
        "id": r[0], "mobile": r[1], "type": r[2], "amount": r[3],
        "status": r[4], "admin_note": r[5], "timestamp": r[6]
    } for r in rows]
    return jsonify({"transactions": data, "limit": limit, "offset": offset})

@bp.post("/transaction/add")
@admin_required
def tx_add():
    data = request.json or {}
    mobile = data.get("mobile")
    typ = data.get("type")
    amount = float(data.get("amount", 0))
    note = data.get("note", "")
    if not (mobile and typ and amount):
        return jsonify({"error": "mobile, type, amount required"}), 400

    # Insert transaction only (does not change balance); use /balance/adjust for wallet changes
    ok, _, wait_log, eta, err = enqueue_write(
        "INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)",
        (mobile, typ, amount, "completed", note)
    )
    _audit("transaction_add", "admin", mobile, f"type={typ}, amount={amount}, note={note}")
    if not ok:
        return jsonify({"error": err or "insert failed", "wait_log": wait_log, "eta": eta}), 400
    return jsonify({"message": "transaction added ✅", "wait_log": wait_log, "eta": eta})

@bp.post("/transaction/refund")
@admin_required
def tx_refund():
    data = request.json or {}
    tx_id = data.get("tx_id")
    note = data.get("note", "refund")
    if not tx_id:
        return jsonify({"error": "tx_id required"}), 400

    # Get tx
    row = run_query("SELECT mobile, amount, type FROM transactions WHERE id=?", (tx_id,), fetch=True)
    if not row:
        return jsonify({"error": "transaction not found"}), 404
    mobile, amount, typ = row[0]

    # Mark original and add reverse entry; credit wallet back
    ok1, _, w1, e1, err1 = enqueue_write(
        "UPDATE transactions SET status='refunded', admin_note=? WHERE id=?",
        (note, tx_id)
    )
    ok2, _, w2, e2, err2 = enqueue_write(
        "INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)",
        (mobile, f"refund_{typ}", amount, "completed", note)
    )
    ok3, _, w3, e3, err3 = enqueue_write(
        "UPDATE users SET balance = balance + ? WHERE mobile=?", (amount, mobile)
    )
    _audit("transaction_refund", "admin", mobile, f"tx_id={tx_id}; amount={amount}; note={note}")
    ok = ok1 and ok2 and ok3
    if not ok:
        return jsonify({"error": err1 or err2 or err3 or "refund failed", "wait_log": w1 + w2 + w3, "eta": max(e1, e2, e3)}), 400
    return jsonify({"message": f"refunded ₹{amount} to {mobile} ✅", "wait_log": w1 + w2 + w3, "eta": max(e1, e2, e3)})

# ---------- Charity / Audit ----------
@bp.get("/charity/balance")
@admin_required
def charity_balance_admin():
    row = run_query("SELECT balance FROM charity_wallet WHERE id=1", fetch=True)
    bal = row[0][0] if row else 0.0
    return jsonify({"charity_balance": bal})

@bp.get("/audit")
@admin_required
def audit_list():
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    rows = run_query(
        "SELECT id, action, actor, target, details, ip, timestamp FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
        fetch=True
    )
    data = [{
        "id": r[0], "action": r[1], "actor": r[2], "target": r[3],
        "details": r[4], "ip": r[5], "timestamp": r[6]
    } for r in rows]
    return jsonify({"audit": data, "limit": limit, "offset": offset})

@bp.post("/cleanup")
@admin_required
def run_cleanup_now():
    from database import cleanup_old_logs
    cleanup_old_logs()
    _audit("cleanup", "admin", "-", "manual")
    return jsonify({"message": "cleanup executed ✅"})
