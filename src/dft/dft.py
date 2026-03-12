from pandas import DataFrame
import pandas as pd


class DFT:
    interpresure: DataFrame = None

    _files = {
        "father": "../dft_data/god_the_father_bible_terms.json",
        "son": "../dft_data/son_bible_terms.json"
    }

    def __init__(self, book: str, chapter: int):
        self.load(book, chapter)

    def load(self, person="son"):
        self.dft = pd.read_json(self._files[person])
        self.dft[['book', 'chapter', 'verse']] = self.dft['verse_reference'].str.extract(r'(\w+) (\d+):(\d+)')


    def _get_file(self, book: str, chapter: int) -> str:
        return self._files[book.lower()][f"{chapter}"]

    def get_annotations(self, book: str, chapter: int = -1) -> DataFrame:
        df = self.dft[self.dft["book"] == book]
        if chapter != -1:
            df = df[df["chapter"] == chapter]
        return df