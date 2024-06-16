import os.path

from .cacher import Cacher
import pandas as pd
from tqdm.auto import tqdm


from .logger import logger
from .utils import BASE_URL
from .event import Event

PE_TABLE_URL = f"{BASE_URL}/posterior/PEtable.txt"
SEARCH_TABLE_URL = f"{BASE_URL}/search/4OGC_top.txt"
SUMMARY_FNAME = f"{Cacher.cache_dir}/ogc_summary.csv"



class Summary():

    def __init__(self):
        self._data = None

    @classmethod
    def load(cls):
        if not os.path.exists(SUMMARY_FNAME):
            return cls._from_ocg()
        return cls._from_cache()

    def download_data(self):
        for index, row in tqdm(self._data.iterrows(), desc="Downloading events data"):
            Event(row['Name']).download_data()


    @classmethod
    def _from_ocg(cls):
        logger.info(f"Getting SUMMARY-TABLE from OGC...")
        s = cls()
        # merge _pe_table + _search_table on "# event" and "Name"
        pe_table = s._pe_table.rename(columns={'# event': 'Name'}, inplace=False)
        data = pd.merge(s._search_table, pe_table, on='Name')
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
            headers = list(pd.read_csv(fpath, sep=" 	 ", skipinitialspace=True).columns.values)
            self.__pe_table = pd.read_csv(fpath, skiprows=1, names=headers)
        return self.__pe_table

    @property
    def _search_table(self):
        if not hasattr(self, "__search_table"):
            fpath = Cacher.get(SEARCH_TABLE_URL)
            headers = list(pd.read_csv(fpath, sep=", ", skipinitialspace=True).columns.values)
            self.__search_table = pd.read_csv(fpath, skiprows=1, names=headers, index_col=0, sep="  ")
        return self.__search_table

    def __len__(self) -> int:
        return 0 if self._data is None else len(self._data)
