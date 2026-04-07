import os
import asyncio
import smtplib
import uvicorn
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# In-memory stores
pending: dict[str, dict] = {}
decisions: dict[str, str] = {}

BASE_URL = os.getenv("HITL_BASE_URL", "http://localhost:8002")

# ── Pydantic model ───────────────────────────────────────────────────────────

class MoveRequest(BaseModel):
    request_id: str
    source: str
    destination_dir: str

# ── Email helper ─────────────────────────────────────────────────────────────

def send_approval_email(request_id: str, source: str, dest: str):
    sender   = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    approve_url = f"{BASE_URL}/decide/{request_id}?approved=true"
    reject_url  = f"{BASE_URL}/decide/{request_id}?approved=false"

    # Plain text fallback
    text_body = f"""
File move approval required (ID: {request_id})

FROM: {source}
TO:   {dest}

Approve: {approve_url}
Reject:  {reject_url}
"""

    # HTML body with clickable buttons
    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2>⚠️ File Move Approval Required</h2>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 8px; font-weight: bold;">Request ID</td>
            <td style="padding: 8px;">{request_id}</td>
        </tr>
        <tr style="background: #f5f5f5;">
            <td style="padding: 8px; font-weight: bold;">FROM</td>
            <td style="padding: 8px; font-family: monospace;">{source}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">TO</td>
            <td style="padding: 8px; font-family: monospace;">{dest}</td>
        </tr>
    </table>
    <br>
    <a href="{approve_url}"
       style="background:#28a745; color:white; padding:10px 20px;
              text-decoration:none; border-radius:4px; margin-right:10px;">
        ✅ Approve
    </a>
    <a href="{reject_url}"
       style="background:#dc3545; color:white; padding:10px 20px;
              text-decoration:none; border-radius:4px;">
        ❌ Reject
    </a>
    <br><br>
    <small style="color: #666;">
        This request will expire after 10 minutes if not actioned.
    </small>
</body>
</html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[File Agent] Approval needed — {request_id}"
    msg["From"]    = sender
    msg["To"]      = receiver

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"📧 Approval email sent to {receiver} (id: {request_id})")
    except Exception as e:
        print(f"❌ Email failed: {e}")
        raise

# ── Auto-expiry helper ───────────────────────────────────────────────────────

async def expire_request(request_id: str, timeout_seconds: int = 600):
    """Auto-reject if human doesn't respond within timeout"""
    await asyncio.sleep(timeout_seconds)
    if request_id in pending:
        pending.pop(request_id)
        decisions[request_id] = "rejected"
        print(f"⏰ Request {request_id} expired — auto-rejected")

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/request-approval")
async def request_approval(req: MoveRequest):
    pending[req.request_id] = {
        "source": req.source,
        "destination_dir": req.destination_dir,
        "status": "pending"
    }

    # Send email notification
    send_approval_email(req.request_id, req.source, req.destination_dir)

    # Start expiry timer in background
    asyncio.create_task(expire_request(req.request_id))

    return {"status": "pending", "request_id": req.request_id}

@app.get("/pending")
async def get_pending():
    """Human can check all pending requests"""
    return pending

@app.get("/decide/{request_id}")
async def decide(request_id: str, approved: bool):
    """
    Human clicks link from email — GET is used here
    so it works directly from email client without a form
    """
    if request_id not in pending and request_id not in decisions:
        return {"error": "Request not found or already decided"}

    if request_id in decisions:
        return {"message": "Already decided", "decision": decisions[request_id]}

    decisions[request_id] = "approved" if approved else "rejected"
    pending.pop(request_id, None)

    action = "✅ Approved" if approved else "❌ Rejected"
    print(f"{action} — request {request_id}")

    # Return a human-readable HTML page
    return_msg = "approved ✅" if approved else "rejected ❌"
    html = f"""
    <html>
    <body style="font-family: Arial; padding: 40px; text-align: center;">
        <h2>Decision recorded</h2>
        <p>Request <strong>{request_id}</strong> has been <strong>{return_msg}</strong>.</p>
        <p>You can close this tab.</p>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

@app.get("/decision/{request_id}")
async def get_decision(request_id: str):
    """Agent polls this endpoint"""
    decision = decisions.get(request_id, "pending")
    return {"decision": decision}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)