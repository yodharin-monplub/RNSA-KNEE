"""Print the most recent messages from the user, regardless of read state."""
import email
import imaplib
import sys
from email.header import decode_header, make_header

from check_email import text_of
from send_email import SENDER_MATCH, credentials

count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
user, password = credentials()
with imaplib.IMAP4_SSL('imap.purelymail.com', 993) as imap:
    imap.login(user, password)
    imap.select('INBOX', readonly=True)
    _, ids = imap.search(None, 'FROM', f'"{SENDER_MATCH}"')
    for num in ids[0].split()[-count:]:
        _, data = imap.fetch(num, '(BODY.PEEK[])')
        message = email.message_from_bytes(data[0][1])
        print('=' * 60)
        print('Date:', message['Date'])
        print('Subject:', str(make_header(decode_header(message['Subject'] or ''))))
        body = text_of(message)
        lines = []
        for line in body.splitlines():
            if line.startswith('>') or (line.startswith('On ') and line.rstrip().endswith('wrote:')):
                break
            lines.append(line)
        print('\n'.join(lines).strip()[:2000])
