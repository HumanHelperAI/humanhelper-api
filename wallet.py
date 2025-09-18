# wallet.py
from writer import enqueue_write
from database import run_query

CHARITY_WITHDRAW_RATE = 0.07  # 7% to charity on withdraw

def deposit(mobile: str, amount: float):
    """
    Returns: (ok, message, wait_log, eta_seconds)
    """
    try:
        sql = "UPDATE users SET balance = balance + ? WHERE mobile=?"
        ok, _, wait_log, eta, err = enqueue_write(sql, (amount, mobile))
        if not ok:
            return False, f"Deposit failed: {err or 'unknown'}", wait_log, eta

        # log transaction with status=completed
        tx_sql = "INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)"
        _, _, wait_log2, eta2, err2 = enqueue_write(tx_sql, (mobile, "deposit", amount, "completed", None))
        wait_log.extend(wait_log2); eta = max(eta, eta2)
        if err2:
            return False, f"Deposit logging failed: {err2}", wait_log, eta

        return True, f"₹{amount} deposited ✅", wait_log, eta
    except Exception as e:
        return False, str(e), [], 0


def withdraw(mobile: str, amount: float):
    """
    Withdraw: apply 7% charity cut on user withdrawal.
    Returns (ok, message, wait_log, eta)
    """
    try:
        rows = run_query("SELECT balance FROM users WHERE mobile=?", (mobile,), fetch=True)
        if not rows:
            return False, "User not found", [], 0
        balance = rows[0][0]
        if balance < amount:
            return False, "Insufficient balance ❌", [], 0

        charity_cut = round(amount * CHARITY_WITHDRAW_RATE, 2)
        user_amount = round(amount - charity_cut, 2)

        # deduct full amount
        sql = "UPDATE users SET balance = balance - ? WHERE mobile=?"
        ok, _, wait_log, eta, err = enqueue_write(sql, (amount, mobile))
        if not ok:
            return False, f"Withdraw failed: {err or 'unknown'}", wait_log, eta

        # add transaction
        tx_sql = "INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)"
        _, _, wait_log2, eta2, err2 = enqueue_write(tx_sql, (mobile, "withdraw", amount, "completed", None))
        wait_log.extend(wait_log2); eta = max(eta, eta2)
        if err2:
            return False, f"Withdraw logging failed: {err2}", wait_log, eta

        # add to charity wallet
        ch_sql = "UPDATE charity_wallet SET balance = balance + ? WHERE id=1"
        _, _, wait_log3, eta3, err3 = enqueue_write(ch_sql, (charity_cut,))
        wait_log.extend(wait_log3); eta = max(eta, eta3)
        if err3:
            return False, f"Charity update failed: {err3}", wait_log, eta

        return True, f"₹{user_amount} sent to user ✅ (₹{charity_cut} → charity)", wait_log, eta
    except Exception as e:
        return False, str(e), [], 0
