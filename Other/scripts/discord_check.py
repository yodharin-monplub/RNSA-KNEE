"""Check Discord for new messages from the user - both the #rnsa channel and direct messages to the bot.

Prints each new human message, oldest first, tagged with where it came from, and remembers the last message id
per source in artifacts/discord_state.json so the same message is not reported twice.
Usage: python scripts/discord_check.py [--all]
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
API = 'https://discord.com/api/v10'
HEADERS = {'Authorization': f"Bot {os.environ['Discord_RNSA_BOT']}"}
STATE = ROOT / 'artifacts/discord_state.json'

sys.path.insert(0, str(ROOT / 'scripts'))
from discord_notify import channel_id, dm_channel_id

state = json.loads(STATE.read_text()) if STATE.exists() else {}
total = 0
# Both places the user can reach me: the #rnsa channel, and a direct message to the bot.
for source, cid in (('#rnsa', channel_id()), ('DM', dm_channel_id())):
    key = f'last_id_{source}'
    params = {'limit': 50}
    if '--all' not in sys.argv and state.get(key):
        params['after'] = state[key]
    messages = requests.get(f'{API}/channels/{cid}/messages', headers=HEADERS, params=params, timeout=30).json()
    if isinstance(messages, dict):                  # an error payload, not a list
        print(f'{source}: discord error: {messages}')
        continue
    human = [m for m in messages if not m.get('author', {}).get('bot')]
    total += len(human)
    for m in sorted(human, key=lambda m: m['id']):
        print(f"[{m['timestamp'][:19]}] {source} {m['author'].get('username')}: {m.get('content', '')}")
    if messages:
        state[key] = max(m['id'] for m in messages)
print(f'unread replies: {total}')
STATE.parent.mkdir(parents=True, exist_ok=True)
STATE.write_text(json.dumps(state))
