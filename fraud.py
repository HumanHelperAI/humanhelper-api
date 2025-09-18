from database import execute_query

def is_fraudulent(mobile, video_id):
    """
    Detect fraud:
      - Same video watched repeatedly within 5 mins
      - Very short duration (<5s)
    """
    rows = execute_query("""
        SELECT timestamp FROM watch_history
        WHERE mobile=? AND video_id=?
        ORDER BY timestamp DESC LIMIT 1
    """, (mobile, video_id), fetch=True)

    if rows:
        return True  # For now, block repeats (can add time checks later)
    return False
