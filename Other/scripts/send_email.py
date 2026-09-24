"""Send a plain-text status email using the ignored email.txt credentials."""
import argparse
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TO = 'yodharinmonplub@actuarialtutor.org'  # AGENTS.md (2026-09-20); earlier mails went to the gmail address
SENDER_MATCH = 'yodharinmonplub'  # IMAP FROM is a substring match: accepts replies from either of the user's addresses


def credentials():
    fields = {}
    for line in (ROOT / 'email.txt').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip().lower()] = value.strip()
    return fields['username'], fields['password']


def send(subject, body):
    user, password = credentials()
    message = EmailMessage()
    # Both of the user's own addresses: AGENTS.md names the actuarialtutor.org one, but the user reads gmail
    # (mails sent only to the former on 2026-09-20 were not seen).
    message['From'], message['To'], message['Subject'] = user, f'{TO}, yodharinmonplub@gmail.com', subject
    message.set_content(body)
    with smtplib.SMTP_SSL('smtp.purelymail.com', 465, context=ssl.create_default_context(), timeout=60) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('body')
    args = parser.parse_args()
    send(args.subject, args.body)
    print('sent')
