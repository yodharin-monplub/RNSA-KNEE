"""Send a message to the user on Discord: the #rnsa channel by default, or a direct message with --dm.

Usage:
    python scripts/discord_notify.py "subject line" [body_file] [--dm]
Token: Discord_RNSA_BOT in Other/.env. Channel: DISCORD_CHANNEL_ID in .env, else the guild's #rnsa channel.
Long messages are split at line boundaries into 1,900-character chunks (Discord's limit is 2,000).
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
API = 'https://discord.com/api/v10'
HEADERS = {'Authorization': f"Bot {os.environ['Discord_RNSA_BOT']}"}
DM_USER_ID = '1193141402658418688'   # yodharin - the only human who has posted in #rnsa


def dm_channel_id():
    """The bot's DM channel with the user. Opening it again returns the same channel, so this is cheap."""
    r = requests.post(f'{API}/users/@me/channels', headers=HEADERS, json={'recipient_id': DM_USER_ID}, timeout=30)
    r.raise_for_status()
    return r.json()['id']


def channel_id():
    if os.environ.get('DISCORD_CHANNEL_ID'):
        return os.environ['DISCORD_CHANNEL_ID']
    guilds = requests.get(f'{API}/users/@me/guilds', headers=HEADERS, timeout=30).json()
    for guild in guilds:
        channels = requests.get(f'{API}/guilds/{guild["id"]}/channels', headers=HEADERS, timeout=30).json()
        text = [c for c in channels if c.get('type') in (0, 5)]
        for c in text:
            if c['name'] == 'rnsa':
                return c['id']
        if text:
            return text[0]['id']
    raise SystemExit('no text channel visible to the bot')


def chunks(text, limit=1900):
    out, current = [], ''
    for line in text.splitlines(keepends=True):
        while len(line) > limit:                      # a single very long line
            out.append(line[:limit]); line = line[limit:]
        if len(current) + len(line) > limit:
            out.append(current); current = ''
        current += line
    if current.strip():
        out.append(current)
    return out


def send(text, cid=None):
    cid = cid or channel_id()
    for part in chunks(text):
        for attempt in range(5):
            r = requests.post(f'{API}/channels/{cid}/messages', headers=HEADERS, json={'content': part}, timeout=30)
            if r.status_code == 429:                  # rate limited: wait what Discord asks for
                time.sleep(float(r.json().get('retry_after', 1)) + 0.5)
                continue
            r.raise_for_status()
            break
        time.sleep(0.4)
    return cid


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--dm']
    subject = args[0]
    body = Path(args[1]).read_text(encoding='utf-8') if len(args) > 1 else ''
    send(f'**{subject}**\n{body}' if body else subject,
         cid=dm_channel_id() if '--dm' in sys.argv else None)
    print('sent')
