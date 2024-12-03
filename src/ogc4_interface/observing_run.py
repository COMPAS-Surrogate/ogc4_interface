"""
https://observing.docs.ligo.org/plan/
https://en.wikipedia.org/wiki/List_of_gravitational_wave_observations

The first run, O1, ran from September 12, 2015, to January 19, 2016, and succeeded in its first gravitational wave detection.
O2 ran for a greater duration, from November 30, 2016, to August 25, 2017.[3]
O3 began on April 1, 2019, which was briefly suspended on September 30, 2019, for maintenance and upgrades, thus O3a.
O3b marks resuming of the run and began on November 1, 2019.
Due to the COVID-19 pandemic[4] O3 was forced to end prematurely.[5]


O4 began on May 24, 2023; initially planned for March, the project needed more time to stabilize the instruments.
The O4 observing run has been extended from one year to 18 months,
following plans to make further upgrades for the O5 run.[2][6]
Updated observing plans are published on the official website, containing the latest information on these runs.[6]
There is a two month commissioning break planned from January to March 2024, after which observations will resume for the remainder of O4.[7]


"""

from datetime import datetime
from typing import List

PERIODS = dict(
    O1=(datetime(2015, 9, 12), datetime(2016, 1, 19)),
    O2=(datetime(2016, 11, 30), datetime(2017, 8, 25)),
    O3a=(datetime(2019, 4, 1), datetime(2019, 10, 1)),
    O3b=(datetime(2019, 11, 1), datetime(2020, 3, 27)),
    O4a=(datetime(2023, 5, 24), datetime(2024, 1, 16)),
    O4b=(datetime(2024, 4, 10), datetime(2025, 6, 9)),
)
DURATIONS = dict(
    O1=5685921,  # https://gwosc.org/timeline/show/O1/H1_DATA*L1_DATA/
    O2=13302397,  # https://gwosc.org/timeline/show/O2/H1_DATA*L1_DATA/
    O3a=11218675,  # https://gwosc.org/timeline/show/O3a_4KHZ_R1/H1_DATA*L1_DATA/
    O3b=9810816,  # https://gwosc.org/timeline/show/O3b_4KHZ_R1/H1_DATA*L1_DATA/
    O4a=None,
    O4b=None,
)


class ObservingRun:
    def __init__(self, name: str):
        self.name = name
        self.start, self.end = PERIODS[name]
        self.duration = DURATIONS[name]

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: str):
        return self.name == other

    def duration(self) -> int:
        return (self.end - self.start).days

    @classmethod
    def from_date(cls, date: datetime) -> "ObservingRun":
        for period, (start, end) in PERIODS.items():
            if start <= date <= end:
                return cls(period)
        raise ValueError(f"Date {date} is not within any observation period")

    @staticmethod
    def get_total_durations(runs: List[str] = None) -> float:
        """In years"""
        if runs is None:
            seconds = sum(DURATIONS.values())
        seconds = sum([DURATIONS[run] for run in runs])
        return seconds / 60 / 60 / 24 / 365.25
