import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AMD_Tools4 as amd
import xml.etree.ElementTree as ET
from io import StringIO
import copy

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

st.title("標高補正付き気象マップ（5mメッシュ + AMD_Tools4）")
st.markdown("標高XMLをアップロードし、AMD気象要素を選んで標高補正分布図を描画します。")

# --- 入力 ---
xml_file = st.file_uploader("📂 5m標高メッシュXMLファイル", type="xml")
element_label = st.selectbox("気象要素を選択", list(ELEMENT_OPTIONS.keys()))
element = ELEMENT_OPTIONS[element_label]
date = st.date_input("対象日を選択")

if st.button("🌏 マップ作成") and xml_file and date:
    try:
        # XML読み込みとパース
        xml_str = xml_file.getvalue().decode("utf-8")
        lines = xml_str.splitlines()
        idx = lines.index('<gml:tupleList>')
        headers = lines[:idx]
        datalist = lines[idx+1:-13]

        body = np.array([float(l.split(',')[1][:-1]) for l in datalist])
        header = lambda tag: next(l for l in headers if tag in l).split(">")[1].split("<")[0].split(" ")

        lats, lons = map(float, header("lowerCorner"))
        late, lone = map(float, header("upperCorner"))
        nola, nolo = [int(x)+1 for x in header("high")[::-1]]

        dlat = (late - lats) / (nola - 1)
        dlon = (lone - lons) / (nolo - 1)
        lat_grid = [lats + dlat * i for i in range(nola)]
        lon_grid = [lons + dlon * j for j in range(nolo)]

        nli50m = body.reshape((nola, nolo))[::-1, :]
        nli50m[nli50m < -990] = np.nan
        lalodomain = [lats, late, lons, lone]

        # --- 気象 & 標高データ取得 ---
        timedomain = [str(date), str(date)]
        Msh, tim, _, _, nam, uni = amd.GetMetData(element, timedomain, lalodomain, namuni=True)
        Msha, _, _, nama, unia = amd.GetGeoData("altitude", lalodomain, namuni=True)

        # --- 形状チェックとデバッグ出力 ---
        st.write(f"気象データ shape: {np.shape(Msh)} / Msh[0]: {np.shape(Msh[0])}")
        st.write(f"標高データ shape: {np.shape(Msha)} / Msha[0]: {np.shape(Msha[0])}")

        # --- 補間処理（全体平均で補間） ---
        def safe_scalar(val, name):
            try:
                return float(val[0])
            except:
                st.warning(f"{name} がスカラーでなかったため、平均値で補間します。shape={np.shape(val)}")
                return float(np.nanmean(val))

        val_msh = safe_scalar(Msh, "気象データ")
        val_msha = safe_scalar(Msha, "標高データ")

        Msh50m = np.full((nola, nolo), val_msh)
        Msha50m = np.full((nola, nolo), val_msha)

        # 標高補正
        corrected = Msh50m + (Msha50m - nli50m) * 0.006

        # --- 図の描画 ---
        st.subheader("🗺️ 標高補正気象マップ")
        figtitle = f"{nam} [{uni}] on {tim[0].strftime('%Y-%m-%d')}"
        tate = 6
        yoko = tate * (max(lon_grid) - min(lon_grid)) / (max(lat_grid) - min(lat_grid)) + 2
        fig = plt.figure(figsize=(yoko, tate))
        plt.axes(facecolor='0.8')

        levels = np.linspace(np.nanmin(corrected), np.nanmax(corrected), 20)
        cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
        cmap.set_over('w', 1.0)
        cmap.set_under('k', 1.0)

        cf = plt.contourf(lon_grid, lat_grid, corrected, levels, cmap=cmap, extend='both')
        plt.colorbar(cf)
        plt.title(figtitle)
        st.pyplot(fig)

        # --- CSV出力 ---
        st.subheader("📥 補正結果のCSVダウンロード")
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
        st.error(f"❌ 処理中にエラーが発生しました: {e}")

elif not xml_file or not date:
    st.info("XMLファイルと日付を指定してください。")