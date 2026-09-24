"""Poll the inbox every 3 minutes; print and exit as soon as the user replies (so the agent is notified)."""
import time
from datetime import datetime

import check_email

while True:
    try:
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            check_email.main()
        text = buffer.getvalue()
        if not text.startswith('unread replies: 0'):
            print(datetime.now().isoformat(timespec='seconds'), flush=True)
            print(text, flush=True)
            break
    except Exception as exc:  # noqa: BLE001 - keep polling through transient IMAP errors
        print('check failed:', type(exc).__name__, str(exc)[:200], flush=True)
    time.sleep(180)
