import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import AMD_Tools4 as amd
import xml.etree.ElementTree as ET
from io import StringIO
import copy
import os

# --- 気象要素の選択肢 ---
ELEMENT_OPTIONS = {
    "日平均気温 (TMP_mea)": "TMP_mea",
    "日最高気温 (TMP_max)": "TMP_max",
    "日最低気温 (TMP_min)": "TMP_min",
    "降水量 (APCP)": "APCP",
    "降水量高精度 (APCPRA)": "APCPRA",
    "降水の有無 (OPR)": "OPR",
    "日照時間 (SSD)": "SSD",
    "全天日射量 (GSR)": "GSR",
    "下向き長波放射量 (DLR)": "DLR",
    "相対湿度 (RH)": "RH",
    "風速 (WIND)": "WIND",
    "積雪深 (SD)": "SD",
    "積雪水量 (SWE)": "SWE",
    "降雪水量 (SFW)": "SFW",
    "予報気温の確からしさ (PTMP)": "PTMP"
}

# --- UI ---
st.title("標高補正付き気象分布マップ作成アプリ")
st.markdown("5mメッシュ標高XMLをアップロードし、AMD_Tools4による気象データと標高補正マップを生成します。")

# --- ファイルアップロード ---
xml_file = st.file_uploader("5mメッシュ標高XMLファイルをアップロード", type="xml")
element_label = st.selectbox("気象要素を選択", list(ELEMENT_OPTIONS.keys()))
element = ELEMENT_OPTIONS[element_label]
date = st.date_input("対象日", value=None)

# --- 実行処理 ---
if st.button("マップ作成") and xml_file and date:
    try:
        # --- XMLをパース ---
        xml_text = xml_file.getvalue().decode("utf-8")
        lines = xml_text.splitlines()
        idx = lines.index('<gml:tupleList>')
        headers = lines[:idx]
        datalist = lines[idx+1:-13]  # フッタ除去（最後13行）

        # 標高値の取得
        num = len(datalist)
        body = np.zeros(num)
        for i in range(num):
            body[i] = float(datalist[i].split(',')[1][:-1])
        nli_raw = body

        # --- ヘッダ情報解析 ---
        def extract_val(tag):
            return next(l for l in headers if tag in l).split(">")[1].split("<")[0].split(" ")

        lower = extract_val("lowerCorner")
        upper = extract_val("upperCorner")
        size = extract_val("high")
        lats, lons = float(lower[0]), float(lower[1])
        late, lone = float(upper[0]), float(upper[1])
        nola, nolo = int(size[1]) + 1, int(size[0]) + 1

        # 緯度経度グリッド作成
        dlat = (late - lats) / (nola - 1)
        dlon = (lone - lons) / (nolo - 1)
        lat_grid = [lats + dlat * i for i in range(nola)]
        lon_grid = [lons + dlon * j for j in range(nolo)]

        # 標高メッシュ
        nli50m = nli_raw.reshape((nola, nolo))[::-1, :]
        nli50m[nli50m < -990] = np.nan
        lalodomain = [lats, late, lons, lone]

        # --- 気象データ・標高データ取得 ---
        timedomain = [str(date), str(date)]
        Msh, tim, _, _, nam, uni = amd.GetMetData(element, timedomain, lalodomain, namuni=True)
        Msha, _, _, nama, unia = amd.GetGeoData("altitude", lalodomain, namuni=True)

        Msh50m = np.full((nola, nolo), Msh[0])
        Msha50m = np.full((nola, nolo), Msha[0])

        # 標高補正
        corrected = Msh50m + (Msha50m - nli50m) * 0.006

        # --- 分布図描画 ---
        st.subheader("📊 補正済み分布図")
        figtitle = f"{nam} [{uni}] on {tim[0].strftime('%Y-%m-%d')}"
        tate = 6
        yoko = tate * (max(lon_grid) - min(lon_grid)) / (max(lat_grid) - min(lat_grid)) + 2
        fig = plt.figure(figsize=(yoko, tate))
        plt.axes(facecolor='0.8')

        levels = np.linspace(np.nanmin(corrected), np.nanmax(corrected), 20)
        cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
        cmap.set_over('w', 1.0)
        cmap.set_under('k', 1.0)
        CF = plt.contourf(lon_grid, lat_grid, corrected, levels, cmap=cmap, extend='both')
        plt.colorbar(CF)
        plt.title(figtitle)
        st.pyplot(fig)

        # --- CSV出力 ---
        st.subheader("📥 CSVダウンロード")
        flat_data = []
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                val = corrected[i, j]
                if not np.isnan(val):
                    flat_data.append([lat, lon, round(val, 3)])
        df = pd.DataFrame(flat_data, columns=["lat", "lon", f"{nam} [{uni}]"])
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSVをダウンロード", csv, file_name="corrected_map.csv", mime="text/csv")

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")

elif not xml_file or not date:
    st.info("XMLファイルと日付を指定してください。")