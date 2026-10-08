"""Update only the RSNA row of Z:/competition.csv; every other byte stays as it is (each line keeps its own terminator -
other sessions write CRLF or LF, sometimes mixed in one file); retry while the file is locked (e.g. open in Excel).
Usage: csv_update.py RANK SPENT PROBABILITY [MAX_MINUTES]"""
import csv
import io
import re
import sys
import time
from pathlib import Path

p = Path('Z:/competition.csv')
rank, spent, prob = sys.argv[1], sys.argv[2], sys.argv[3]
deadline = time.time() + 60 * float(sys.argv[4] if len(sys.argv) > 4 else 120)
NAME = 'RSNA Knee Abnormality Detection (Kaggle),'
while True:
    try:
        raw = p.read_bytes()
        text = raw.decode('utf-8')
        parts = re.split(r'(\r\n|\n)', text)          # [line, eol, line, eol, ..., last]
        lines, eols = parts[0::2], parts[1::2]
        header = next(csv.reader([lines[0]]))
        idx = [i for i, l in enumerate(lines) if l.startswith(NAME)]
        assert len(idx) == 1, idx
        old = next(csv.reader([lines[idx[0]]]))
        new = [old[0], rank, old[2], spent] + ([prob] if len(header) >= 5 else [])
        new += [''] * (len(header) - len(new))
        buf = io.StringIO(); csv.writer(buf, lineterminator='').writerow(new)
        lines[idx[0]] = buf.getvalue()
        out = ''.join(l + (eols[i] if i < len(eols) else '') for i, l in enumerate(lines))
        p.write_bytes(out.encode('utf-8'))
        check = re.split(r'(\r\n|\n)', p.read_bytes().decode('utf-8'))
        others_ok = all(a == b for i, (a, b) in enumerate(zip(parts, check)) if i != 2 * idx[0])
        print(time.strftime('%H:%M'), 'UPDATED |', check[2 * idx[0]], '| other rows and line ends unchanged:', others_ok,
              flush=True)
        break
    except PermissionError:
        if time.time() > deadline:
            print('TIMEOUT: Z:/competition.csv stayed locked', flush=True); sys.exit(1)
        time.sleep(120)
