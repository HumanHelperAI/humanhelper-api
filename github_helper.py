# github_helper.py
import os
import requests
from typing import Tuple, List, Dict
from flask import Blueprint, request, jsonify

gh_bp = Blueprint("github", __name__, url_prefix="/github")

@gh_bp.get("/repos")
def list_repos():
    # ...
    return jsonify(...)

@gh_bp.post("/issue")
def create_issue():
    # ...
    return jsonify(...)
# get token from env
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "") or os.getenv("GITHUB_API_TOKEN", "")

API_BASE = "https://api.github.com"

def _auth_headers():
    if not GITHUB_TOKEN:
        return {}
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "HumanHelperAI/1.0"
    }

def gh_get_user() -> Tuple[bool, Dict]:
    """Return (ok, json_or_error) for the token owner"""
    if not GITHUB_TOKEN:
        return False, {"error": "GITHUB_TOKEN not set"}
    try:
        r = requests.get(f"{API_BASE}/user", headers=_auth_headers(), timeout=10)
        if r.status_code >= 400:
            return False, {"status": r.status_code, "body": r.text}
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def gh_list_repos(per_page: int = 100) -> Tuple[bool, List[Dict]]:
    """Return list of repos visible to token owner (paginated single page)"""
    if not GITHUB_TOKEN:
        return False, {"error": "GITHUB_TOKEN not set"}
    try:
        params = {"per_page": per_page}
        r = requests.get(f"{API_BASE}/user/repos", headers=_auth_headers(), params=params, timeout=10)
        if r.status_code >= 400:
            return False, {"status": r.status_code, "body": r.text}
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def gh_create_issue(full_repo_name: str, title: str, body: str = "", labels: list | None = None) -> Tuple[bool, Dict]:
    """
    Create an issue.
      full_repo_name = "owner/repo"
      title = issue title
      body = issue body/description
      labels = optional list of labels
    Returns (ok, response_json_or_error)
    """
    if not GITHUB_TOKEN:
        return False, {"error": "GITHUB_TOKEN not set"}
    if "/" not in full_repo_name:
        return False, {"error": "repo must be owner/repo"}
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    try:
        r = requests.post(f"{API_BASE}/repos/{full_repo_name}/issues", headers=_auth_headers(), json=payload, timeout=10)
        if r.status_code >= 400:
            # include helpful error text
            try:
                return False, {"status": r.status_code, "body": r.json()}
            except Exception:
                return False, {"status": r.status_code, "body": r.text}
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

# ------- Flask endpoints (simple wrappers) -------
# Note: protecting create-issue with admin token is done at registration site (app.py)
@gh_bp.get("/user")
def _route_get_user():
    ok, out = gh_get_user()
    if not ok:
        return jsonify({"ok": False, "error": out}), 400
    return jsonify({"ok": True, "user": out})

@gh_bp.get("/repos")
def _route_list_repos():
    ok, out = gh_list_repos()
    if not ok:
        return jsonify({"ok": False, "error": out}), 400
    # reduce to useful fields to avoid huge payloads
    simplified = [{"full_name": r.get("full_name"), "private": r.get("private"), "html_url": r.get("html_url")} for r in out]
    return jsonify({"ok": True, "repos": simplified})

@gh_bp.post("/issue")
def _route_create_issue():
    data = request.json or {}
    repo = data.get("repo") or data.get("full_repo") or data.get("full_name")
    title = data.get("title") or ""
    body = data.get("body") or data.get("description") or ""
    labels = data.get("labels") or None

    if not repo or not title:
        return jsonify({"ok": False, "error": "repo and title are required (repo should be owner/repo)"}), 400

    ok, out = gh_create_issue(repo, title, body, labels=labels)
    if not ok:
        return jsonify({"ok": False, "error": out}), 400
    return jsonify({"ok": True, "issue": out}), 201
