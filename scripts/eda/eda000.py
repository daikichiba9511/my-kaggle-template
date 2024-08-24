import IPython.display as ipd
import polars as pl

from src import constants, log

pl.Config.set_tbl_cols(50)

logger = log.get_root_logger()

df = pl.read_csv(constants.DATA_DIR / "train.csv")
ipd.display(df)
print("** DONE **")
