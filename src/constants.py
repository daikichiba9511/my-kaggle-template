import os
import pathlib

IS_KAGGLE = os.getenv("KAGGLE") is not None
COMPE_NAME = "isic-2024-challenge"

ROOT = pathlib.Path("/kaggle") if IS_KAGGLE else pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = INPUT_DIR / COMPE_NAME


ID_COL = "isic_id"
TARGET_COL = "target"
NUM_COLS = [
    "clin_size_long_diam_mm",  # 病変の最大直径 (mm)
    "tbp_lv_A",  # 内側のA病変+
    "tbp_lv_Aext",  # 外側のA病変+
    "tbp_lv_B",  # 内側のB病変+
    "tbp_lv_Bext",  # 外側のB病変+
    "tbp_lv_C",  # 病変内部の彩度+
    "tbp_lv_Cext",  # 病変外部の彩度+
    "tbp_lv_H",  # 病変内部の色合い, LAB*色空間におけるA*とB*の角度. 一般的なレンジは25(赤)から75(茶色)
    "tbp_lv_Hext",  # 病変外部の色合い
    "tbp_lv_L",  # L病変内部
    "tbp_lv_Lext",  # L病変外部
    "tbp_lv_areaMM2",  # 病変の面積 (mm^2)
    "tbp_lv_area_perim_ratio",  # 病変の面積と周囲長の比率, 境界のギザギザ度合いを示す, 円形では低く、不規則な形状では高い
    "tbp_lv_color_std_mean",  # 色の不規則性、病変境界内の色の分散として計算される
    "tbp_lv_deltaA",  # 平均Aコントラスト(病変の内側と外側)
    "tbp_lv_deltaB",  # 平均Bコントラスト(病変の内側と外側)
    "tbp_lv_deltaL",  # 平均Lコントラスト(病変の内側と外側)
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",  # 病変とその周囲の皮膚とのコントラスト、コントラストの高い病変は色素が濃い傾向がある。LAB色空間における直接の背景に対する病変の平均deltaLB
    "tbp_lv_eccentricity",  # 病変の偏心率
    "tbp_lv_minorAxisMM",  # 最小病変直径 (mm)
    "tbp_lv_nevi_confidence",  # CNNで病変が母斑として予測される確信度
    "tbp_lv_norm_border",  # 病変の境界の不規則性(0-10scale).境界のギザギザと非対称性の正規化された平均
    "tbp_lv_norm_color",  # カラーバリエーション。色の非対称性と色むらの正規化された平均
    "tbp_lv_perimeterMM",  # 病変の周囲長 (mm)
    "tbp_lv_radial_color_std_max",  # 色の非対称性。病変内の色の空間分布の非対称性の尺度。
    "tbp_lv_stdL",  # L病変内の標準偏差
    "tbp_lv_stdLExt",  # L病変外部の標準偏差
    "tbp_lv_symm_2axis",  # 境界線の非対称性。病変の最も対称的な軸に垂直な軸を中心とした病変の輪郭の非対称性の尺度
    "tbp_lv_symm_2axis_angle",  # 病変境界非対称性の角度
    "tbp_lv_x",  # 3D TBPのX座標
    "tbp_lv_y",  # 3D TBPのY座標
    "tbp_lv_z",  # 3D TBPのZ座標
    "tbp_lv_dnn_lesion_confidence",  # DNNで病変が存在すると予測される確信度
]

STR_COLS = [
    # "isic_id",
    # "patient_id",
    "age_approx",  # 年齢、NAがあるからstrになってる
    "sex",
    "anatom_site_general",  # 解剖学的部位
    "image_type",
    "tbp_tile_type",  # TBPのタイルタイプ
    "tbp_lv_location",  # TBPの位置
    "tbp_lv_location_simple",  # TBPの位置の簡易版
    "attribution",
    "copyright_license",
    "lesion_id",
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "mel_mitotic_index",
    "mel_thick_mm",
]
