"""Re-extract the twelve knee findings from radiology reports with an LLM (OpenRouter).

Usage: python scripts/llm_relabel.py --subset gold --model anthropic/claude-sonnet-5 --tag s5v1
Writes data/llm_relabel/<tag>.csv (probabilities) and a jsonl cache for resumability.
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

PROMPT = """You are a musculoskeletal radiologist. Below is a knee MRI report (it may be in any language).
Expert radiologists independently re-read the MRI images and marked each of 12 findings as present or absent.
Estimate, for each finding, the probability (0 to 1) that the expert image read marked it PRESENT.

Definitions used by the image readers:
- ACL: anterior cruciate ligament tear (partial or complete, including high-grade sprain / mucoid degeneration only if described as torn).
- MCL: medial collateral ligament injury (sprain grade 1-3 or tear).
- Medial Meniscus / Lateral Meniscus: meniscal tear of that meniscus (any type, including degenerative tear, maceration, extrusion with tear). Intrasubstance degeneration without surface contact is NOT a tear.
- Medial OA / Lateral OA / PF OA: osteoarthritis of the medial tibiofemoral, lateral tibiofemoral, or patellofemoral compartment (cartilage loss / chondromalacia of meaningful grade, osteophytes, subchondral changes).
- Effusion: joint effusion clearly more than physiologic fluid (small/mild effusions may be judged physiologic by the readers).
- Synovitis: synovial thickening/inflammation, including Hoffa fat pad synovitis; readers mark it when an effusion-synovitis complex is visible even if the report does not say "synovitis".
- Baker's: popliteal (Baker's) cyst of any size.
- Contusion: bone contusion / bone bruise / trabecular microfracture pattern of bone marrow edema from trauma (not degenerative subchondral edema).
- Fracture: any fracture line, including osteochondral, subchondral insufficiency, avulsion, or impaction fracture.
Findings not mentioned are usually absent, but consider what an image reader would likely see given the whole report
(e.g., advanced OA often accompanies effusion and synovitis; acute ACL tears usually have contusions).

Return ONLY a JSON object with exactly these keys and numeric probabilities:
{keys}

REPORT:
{report}"""


# v2: the host's published annotation criteria (docs/official_label_criteria.md) instead of our own definitions.
PROMPT_V2 = """You are a musculoskeletal radiologist. Below is a knee MRI report (it may be in any language).
Two subspecialty MSK radiologists independently re-read the MRI IMAGES (without this report) and marked each of
12 findings present or absent; a third adjudicated. Estimate, for each finding, the probability (0 to 1) that the
consensus image read marked it PRESENT.

Criteria the image readers used (borderline / "on the fence" findings were graded NEGATIVE):
- ACL: high-grade partial (>50% of fibres) or full-thickness ACL tear. Mild signal change, mucoid degeneration, low-grade sprain or thickening without discontinuity is negative.
- MCL: high-grade partial or complete ACUTE MCL tear with disrupted fibres and edema. Low-grade (grade 1) sprains and chronic/remote changes are negative.
- Medial Meniscus / Lateral Meniscus: tear of that meniscus = signal definitely reaching the surface on at least two images, or truncated / diminutive / displaced fragment. Intrasubstance degeneration is negative.
- Medial OA / Lateral OA / PF OA: a moderate or large area (about 1 cm or more) of high-grade cartilage loss (>50% thickness) in that compartment, with or without subchondral marrow change. Low-grade chondromalacia, small focal defects or osteophytes alone are negative.
- Effusion: a MODERATE or LARGE amount of joint fluid. Physiologic, trace, minimal is negative; "small"/"mild" is borderline.
- Synovitis: inflammation and thickening of the synovial lining.
- Baker's: a MODERATE or LARGE popliteal (Baker) cyst. Tiny/trace/small cysts are usually negative.
- Contusion: bone marrow edema-like signal from impact (bone bruise) without a discrete fracture line. Degenerative subchondral edema is not a contusion.
- Fracture: an ACUTE cortical break or fracture line. Old/healed fractures are negative.

Calibration guidance: size words in reports are noisy (one reader's "small" is another's "moderate"), so borderline
wording deserves a middling probability rather than 0. A report that explicitly denies a finding is stronger
evidence of absence than a report that never mentions it; a finding the report never mentions is usually absent,
but consider what image readers would likely see given the whole report (e.g. acute ACL tears usually come with
contusions and effusion; advanced OA often comes with effusion and synovitis).

Return ONLY a JSON object with exactly these keys and numeric probabilities:
{keys}

REPORT:
{report}"""

PROMPTS = {'v1': PROMPT, 'v2': PROMPT_V2}
ACTIVE = {'prompt': PROMPT}

EXTRA = {}


def call(model, report, key):
    body = {'model': model, 'temperature': 0, 'max_tokens': 1500, **EXTRA,
            'messages': [{'role': 'user', 'content': ACTIVE['prompt'].format(keys=json.dumps({k: 0.0 for k in LABELS}), report=report)}]}
    for attempt in range(9):
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions', timeout=120,
                              headers={'Authorization': 'Bearer ' + key}, json=body)
            r.raise_for_status()
            data = r.json()
            choice = data['choices'][0]
            text = choice['message'].get('content')
            if not text:
                raise ValueError(f"empty content, finish={choice.get('finish_reason')}")
            obj = json.loads(re.search(r'\{.*\}', text, re.S).group(0))
            probs = {k: min(1.0, max(0.0, float(obj[k]))) for k in LABELS}
            return probs, data.get('usage', {})
        except Exception as exc:  # noqa: BLE001 - retry any transient/parsing failure
            if attempt == 8:
                print('LLM call failed:', type(exc).__name__, str(exc)[:200], flush=True)
                return None, {}
            time.sleep(min(60, 2 ** attempt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', choices=['gold', 'all'], default='gold')
    p.add_argument('--model', default='anthropic/claude-sonnet-5')
    p.add_argument('--tag', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no-reasoning', action='store_true')
    p.add_argument('--effort')
    p.add_argument('--prompt', choices=sorted(PROMPTS), default='v1')
    args = p.parse_args()
    ACTIVE['prompt'] = PROMPTS[args.prompt]
    if args.no_reasoning:
        EXTRA['reasoning'] = {'effort': 'minimal'}
    if args.effort:
        EXTRA['reasoning'] = {'effort': args.effort}
    key = os.environ['OPENROUTER_API_KEY']
    train = pd.read_csv(ROOT / 'data/train.csv')
    if args.subset == 'gold':
        train = train.dropna(subset=LABELS)
    out = ROOT / 'data/llm_relabel'
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f'{args.tag}.jsonl'
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done[row['StudyInstanceUID']] = row
    lock = threading.Lock()
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    todo = [(r.StudyInstanceUID, r.Report) for r in train.itertuples() if r.StudyInstanceUID not in done]
    print('todo', len(todo), 'cached', len(done), flush=True)

    def work(item):
        uid, report = item
        probs, u = call(args.model, str(report), key)
        if probs is None:
            return
        row ={'StudyInstanceUID': uid, **probs}
        with lock:
            done[uid] = row
            for k in usage:
                usage[k] += int(u.get(k, 0) or 0)
            with cache_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            if len(done) % 100 == 0:
                print('done', len(done), usage, flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    frame = pd.DataFrame([done[u] for u in train.StudyInstanceUID if u in done])
    frame.to_csv(out / f'{args.tag}.csv', index=False)
    print('wrote', len(frame), 'usage', usage, flush=True)


if __name__ == '__main__':
    main()
