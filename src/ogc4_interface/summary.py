import os.path
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .cacher import BASE_URL, Cacher
from .event import Event
from .logger import logger
from .plotting import plot_scatter

PE_TABLE_URL = f"{BASE_URL}/posterior/PEtable.txt"
SEARCH_TABLE_URL = f"{BASE_URL}/search/4OGC_top.txt"
SUMMARY_FNAME = f"{Cacher.cache_dir}/ogc_summary.csv"


class Summary:
    def __init__(self):
        self._data = None

    @classmethod
    def load(cls):
        if not os.path.exists(SUMMARY_FNAME):
            return cls._from_ocg()
        return cls._from_cache()

    def download_data(self):
        for index, row in tqdm(
            self._data.iterrows(), desc="Downloading events data"
        ):
            Event(row["Name"]).download_data()

    @classmethod
    def _from_ocg(cls):
        logger.info(f"Getting SUMMARY-TABLE from OGC...")
        s = cls()
        # merge _pe_table + _search_table on "# event" and "Name"
        pe_table = s._pe_table.rename(
            columns={"# event": "Name"}, inplace=False
        )
        data = pd.merge(s._search_table, pe_table, on="Name")
        data.to_csv(SUMMARY_FNAME, index=False)
        s._data = data
        return s

    @classmethod
    def _from_cache(cls):
        logger.info(f"Getting SUMMARY-TABLE from cache...")
        s = cls()
        s._data = pd.read_csv(SUMMARY_FNAME)
        return s

    @property
    def _pe_table(self):
        if not hasattr(self, "__pe_table"):
            fpath = Cacher.get(PE_TABLE_URL)
            headers = list(
                pd.read_csv(
                    fpath, sep=" 	 ", skipinitialspace=True
                ).columns.values
            )
            self.__pe_table = pd.read_csv(fpath, skiprows=1, names=headers)
        return self.__pe_table

    @property
    def _search_table(self):
        if not hasattr(self, "__search_table"):
            fpath = Cacher.get(SEARCH_TABLE_URL)
            headers = list(
                pd.read_csv(
                    fpath, sep=", ", skipinitialspace=True
                ).columns.values
            )
            self.__search_table = pd.read_csv(
                fpath, skiprows=1, names=headers, index_col=0, sep="  "
            )
        return self.__search_table

    def __len__(self) -> int:
        return 0 if self._data is None else len(self._data)

    def get_pastro_threholded_event_names(self, pastro_threshold) -> List[str]:
        return self.get_filtered_data(pastro_threshold)["Name"].values

    def get_filtered_data(self, pastro_threshold: float):
        return self._data[self._data["Pastro"] >= pastro_threshold]

    def plot(self, ax=None, pastro_threshold=0.95, bounds=None, color="k"):
        d = self.get_filtered_data(pastro_threshold)[
            ["redshift", "srcmchirp"]
        ].values
        ax = plot_scatter(d, ax=ax, bounds=bounds, color=color)
        return ax

    def get_mcz_for(self, event_name: str):
        return self._data[self._data["Name"] == event_name][
            ["srcmchirp", "redshift"]
        ].values[0]

    @property
    def event_names(self):
        return self._data["Name"].values

    @property
    def data_records(self) -> np.ndarray:
        d = self._data[
            [
                "Name",
                "srcmchirp",
                "srcmchirp_plus",
                "srcmchirp_minus",
                "redshift",
                "redshift_plus",
                "redshift_minus",
                "Pastro",
            ]
        ]
        d["Name"] = d["Name"].astype("S15")
        return d.to_records(index=False)
