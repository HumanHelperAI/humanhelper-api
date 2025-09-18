
# earnings.py
from writer import enqueue_write
from database import run_query

def reward_user(mobile, video_id, content_type, duration):
    """
    Returns (ok, result) where result is either error message or dict {"earned": x, "charity": y}
    """
    # Fraud prevention: same video once per user
    watched = run_query("SELECT id FROM watch_history WHERE mobile=? AND video_id=?", (mobile, video_id), fetch=True)
    if watched:
        return False, "Already watched ❌ No reward"

    if content_type == "video":
        minutes = duration / 60.0
        reward = round(0.15 * minutes, 2)
        charity = round(0.03 * minutes, 2)
    elif content_type == "short":
        reward = 0.03
        charity = 0.01
    else:
        return False, "Invalid content type ❌"

    user_reward = round(reward - charity, 2)

    # Update user wallet (via writer)
    ok1, _, l1, e1, err1 = enqueue_write("UPDATE users SET balance = balance + ? WHERE mobile=?", (user_reward, mobile))
    if not ok1:
        return False, f"Failed to credit user: {err1 or 'unknown'}"

    # Log transaction
    ok2, _, l2, e2, err2 = enqueue_write("INSERT INTO transactions (mobile, type, amount, status, admin_note) VALUES (?,?,?,?,?)",
                                         (mobile, "earn", user_reward, "completed", None))
    if not ok2:
        return False, f"Failed to log transaction: {err2 or 'unknown'}"

    # Update charity
    ok3, _, l3, e3, err3 = enqueue_write("UPDATE charity_wallet SET balance = balance + ? WHERE id=1", (charity,))
    if not ok3:
        return False, f"Failed to update charity: {err3 or 'unknown'}"

    # Save history
    ok4, _, l4, e4, err4 = enqueue_write(
        "INSERT INTO watch_history (mobile, video_id, content_type, duration, reward, charity) VALUES (?,?,?,?,?,?)",
        (mobile, video_id, content_type, duration, user_reward, charity)
    )
    if not ok4:
        return False, f"Failed to save watch_history: {err4 or 'unknown'}"

    return True, {"earned": user_reward, "charity": charity}
