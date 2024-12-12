import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from src import constants, log, metrics, utils

from . import config, preprocess

logger = log.get_root_logger()
CALLED_TIME = log.get_called_time()
COMMIT_HASH = utils.get_commit_hash_head()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(debug: bool = False, seed: int = 42) -> None:
    cfg = config.GBDTConfig(seed=seed, is_debug=debug)

    fname = __file__.split("/")[-1].split(".")[0]
    save_dir = cfg.output_dir / f"{fname}/{cfg.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.attach_file_handler(logger, str(save_dir / "train_gbdt.log"))

    df = pl.read_parquet(cfg.train_data_fp)
    # -- Train
    oof = []
    for fold in range(cfg.n_folds):
        # -- Make fold
        df_train = df.filter(pl.col("fold") != fold)
        df_valid = df.filter(pl.col("fold") == fold)
        # -- Preprocess
        df_train, df_valid = preprocess.apply_preprocess(df_train, df_valid)

        cols_feature = [col for col in df.columns if col not in ["fold", constants.COL_TARGET, cfg.col_key]]
        cols_categorical = [col for col in cols_feature if df[col].dtype == pl.Categorical]

        y_train = df_train[constants.COL_TARGET].to_numpy()
        ds_train = lgb.Dataset(
            df_train.select(cols_feature).to_pandas(),
            label=y_train,
            feature_name=cols_feature,
            categorical_feature=cols_categorical,
            free_raw_data=False,
        )
        ds_valid = lgb.Dataset(
            df_valid.select(cols_feature).to_pandas(),
            label=df_valid[constants.COL_TARGET].to_numpy(),
            feature_name=cols_feature,
            categorical_feature=cols_categorical,
            reference=ds_train,
            free_raw_data=False,
        )
        model = lgb.train(
            params=cfg.gbdt_model_params,
            train_set=ds_train,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[ds_train, ds_valid],
            valid_names=["train", "valid"],
            callbacks=[lgb.log_evaluation(period=500)],
        )
        model.save_model(save_dir / f"xgb_model_{fold}.ubj")
        cols = {"cols_feature": cols_feature, "cols_categorical": cols_categorical}
        utils.save_as_json(cols, save_dir / f"cols_{fold}.json")

        y_pred_raw = model.predict(ds_valid.data)
        assert isinstance(y_pred_raw, np.ndarray)
        y_pred = y_pred_raw
        print(df_valid)
        oof_fold = pl.DataFrame({
            "id": df_valid[cfg.col_key].to_numpy().reshape(-1),
            "fold": [fold] * len(y_pred),
            "y_true": df_valid[constants.COL_TARGET].to_numpy().reshape(-1),
            "y_pred": y_pred.reshape(-1),
        })
        print(oof_fold)

        oof_fold.write_parquet(save_dir / f"oof_{fold}.parquet")
        oof.append(oof_fold)

        importances = model.feature_importance(importance_type="gain")
        feature_cols = model.feature_name()
        df_importances = pl.DataFrame({"feature": feature_cols, "gain": importances})
        df_importances.write_parquet(save_dir / f"importances_fold{fold}_{cfg.seed}.parquet")
        utils.save_importances(df_importances, save_dir / f"importances_fold{fold}_{cfg.seed}.png")

    oof = pl.concat(oof)
    oof.write_parquet(save_dir / f"best_oof_{cfg.seed}.parquet")
    score_oof = metrics.score(y_pred=oof["y_pred"].to_numpy(), y_true=oof["y_true"].to_numpy())
    logger.info(f"OOF Score: {score_oof}")
