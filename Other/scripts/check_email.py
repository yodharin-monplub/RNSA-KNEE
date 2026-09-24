"""Print unread replies from the user (IMAP, same credentials as send_email.py); marks them read."""
import email
import imaplib
from email.header import decode_header, make_header

from send_email import SENDER_MATCH, credentials


def text_of(message):
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == 'text/plain' and not part.get_filename():
                return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'replace')
        return ''
    return message.get_payload(decode=True).decode(message.get_content_charset() or 'utf-8', 'replace')


def main():
    user, password = credentials()
    with imaplib.IMAP4_SSL('imap.purelymail.com', 993) as imap:
        imap.login(user, password)
        imap.select('INBOX')
        _, ids = imap.search(None, 'UNSEEN', 'FROM', f'"{SENDER_MATCH}"')
        found = ids[0].split()
        print('unread replies:', len(found))
        for num in found:
            _, data = imap.fetch(num, '(RFC822)')
            message = email.message_from_bytes(data[0][1])
            print('=' * 60)
            print('Date:', message['Date'])
            print('Subject:', str(make_header(decode_header(message['Subject'] or ''))))
            body = text_of(message)
            # Drop quoted history from the reply.
            lines = []
            for line in body.splitlines():
                if line.startswith('>') or (line.startswith('On ') and line.rstrip().endswith('wrote:')):
                    break
                lines.append(line)
            print('\n'.join(lines).strip()[:3000])


if __name__ == '__main__':
    main()
