"""Download the MIT-BIH Arrhythmia Database (about 73 MB) from PhysioNet into data/."""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, MITDB  # noqa: E402

URL = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"

if MITDB.exists() and any(MITDB.glob("*.dat")):
    print("already downloaded:", MITDB)
    sys.exit(0)

DATA.mkdir(exist_ok=True)
zpath = DATA / "mitdb.zip"
print("downloading", URL)
urllib.request.urlretrieve(URL, zpath)
with zipfile.ZipFile(zpath) as z:
    z.extractall(DATA)
zpath.unlink()
print(len(list(MITDB.glob("*.dat"))), "records in", MITDB)
