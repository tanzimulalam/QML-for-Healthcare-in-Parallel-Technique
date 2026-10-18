"""Paths and dataset constants used by all the scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MITDB = DATA / "mit-bih-arrhythmia-database-1.0.0"
# QML_RESULTS (relative to the repository) lets scripts/quick_test.sh write
# somewhere else, so a test run never overwrites the real results
RESULTS = ROOT / os.environ.get("QML_RESULTS", "results")

FS = 360        # MIT-BIH sampling rate in Hz
HALF_WIN = 90   # samples each side of the R peak, so every beat is a 0.5 s window

# Inter-patient split from de Chazal et al. (2004). The paced records
# 102, 104, 107 and 217 are not used.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# annotation symbol -> AAMI class (N=0, S=1, V=2, F=3); Q beats are dropped
AAMI = {**dict.fromkeys("NLRej", 0), **dict.fromkeys("AaJS", 1),
        **dict.fromkeys("VE", 2), "F": 3}
CLASSES = ["N", "S", "V", "F"]
V = 2
