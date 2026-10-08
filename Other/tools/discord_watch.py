"""Exit (and so notify the session) as soon as the user sends a Discord message; checks every 3 minutes.
Usage: python discord_watch.py [MAX_HOURS]"""
import subprocess, sys, time
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
end = time.time() + 3600 * float(sys.argv[1] if len(sys.argv) > 1 else 24)
while time.time() < end:
    try:
        r = subprocess.run([str(OTHER / '.venv/Scripts/python.exe'), str(OTHER / 'scripts/discord_check.py')],
                           capture_output=True, text=True, timeout=120)
        out = r.stdout.strip()
    except Exception as exc:
        out = f'check error {type(exc).__name__}'
    last = (out.splitlines() or [''])[-1]
    if last.startswith('unread replies:') and last != 'unread replies: 0':
        print(time.strftime('%m-%d %H:%M'), 'EVENT discord |', out[-3000:], flush=True)
        sys.exit(0)
    time.sleep(180)
print('TIMEOUT', time.strftime('%m-%d %H:%M'), flush=True)
