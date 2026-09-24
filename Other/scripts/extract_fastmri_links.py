"""Save the fastMRI download links from the user's forwarded NYU email to data/external/fastmri/links.json.

The links are signed URLs: they are written to an ignored data folder and never printed.
"""
import email
import imaplib
import json
import re
from html import unescape
from pathlib import Path

from send_email import TO, credentials

ROOT = Path(__file__).resolve().parents[1]
user, password = credentials()
with imaplib.IMAP4_SSL('imap.purelymail.com', 993) as imap:
    imap.login(user, password)
    imap.select('INBOX', readonly=True)
    _, ids = imap.search(None, 'FROM', f'"{TO}"', 'SUBJECT', '"FastMRI"')
    num = ids[0].split()[-1]
    _, data = imap.fetch(num, '(BODY.PEEK[])')
message = email.message_from_bytes(data[0][1])
links = {}
for part in message.walk():
    if part.get_content_type() not in ('text/html', 'text/plain'):
        continue
    text = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'replace')
    if part.get_content_type() == 'text/html':
        for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S | re.I):
            name = re.sub(r'<[^>]+>', '', unescape(label)).strip().split()[0] if label.strip() else ''
            if name:
                links.setdefault(name, unescape(href))
    else:
        # Gmail's plain-text forward wraps long URLs; rejoin them, then key each URL by the archive it names.
        joined = re.sub(r'=?\r?\n(?=\S)(?<!\n\n)', '', text) if False else text
        for href in re.findall(r'https?://[^\s"<>]+(?:\r?\n[^\s"<>]+)*?(?=["\s<>])', joined):
            href = re.sub(r'\s+', '', href)
            match = re.search(r'/((?:knee|brain|fastMRI|fastmri|labels)[\w.-]*?)\.(?:tar\.xz|tar\.gz|zip|tar)', href)
            if match:
                links.setdefault(match.group(1), href)
out = ROOT / 'data/external/fastmri'
out.mkdir(parents=True, exist_ok=True)
(out / 'links.json').write_text(json.dumps(links, indent=1))
knee = sorted(k for k in links if k.lower().startswith('knee'))
print('links saved:', len(links), '| knee entries:', knee)
print('hosts:', sorted({re.sub(r'^https?://([^/]+)/.*$', r'\1', v) for v in links.values()}))
