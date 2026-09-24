import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.data import make_partitions

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    frame = make_partitions(root / 'data', root / 'artifacts/partitions')
    print(frame.split.value_counts().to_string())
