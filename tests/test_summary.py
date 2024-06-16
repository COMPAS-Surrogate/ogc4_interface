from ogc4_interface.summary import Summary, SUMMARY_FNAME
import os

def test_summary():
    if os.path.exists(SUMMARY_FNAME):
        os.remove(SUMMARY_FNAME)
    summary = Summary.load() # via the OGC
    summary = Summary.load() # via the cache
    assert len(summary) == 94
    summary.download_data()