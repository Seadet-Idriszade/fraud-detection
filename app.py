import json
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sığorta fraud radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Canlı kiber-radar və aurora vizual dizaynı
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #060913 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.5) 0px, transparent 100%) !important;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: -100px;
        left: -100px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 70%);
        filter: blur(90px);
        border-radius: 50%;
        z-index: 0;
        pointer-events: none;
        animation: floatOrb 20s ease-in-out infinite alternate;
    }

    .stApp::after {
        content: "";
        position: fixed;
        bottom: -120px;
        right: -100px;
        width: 550px;
        height: 550px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.16), transparent 70%);
        filter: blur(100px);
        border-radius: 50%;
        z-index: 0;
        pointer-events: none;
        animation: floatOrb 25s ease-in-out infinite alternate-reverse;
    }

    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(80px, 50px) scale(1.1); }
        100% { transform: translate(-40px, 90px) scale(0.95); }
    }

    .custom-card, div[data-testid="stMetric"], .stExpander, div[data-testid="stDataFrame"] {
        background: rgba(13, 20, 36, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6) !important;
        transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.3s, box-shadow 0.3s !important;
    }

    .custom-card:hover, div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 34px -8px rgba(56, 189, 248, 0.15) !important;
    }

    .radar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
        margin-top: -15px;
    }

    .radar-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #ffffff 15%, #38bdf8 55%, #a855f7 90%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 8s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .radar-dot {
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 14px rgba(56, 189, 248, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
        color: #f8fafc !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        padding: 0 18px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15) !important;
    }

    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-danger { 
        background: rgba(244, 63, 94, 0.15); 
        color: #fb7185; 
        border: 1px solid rgba(244, 63, 94, 0.4);
        box-shadow: 0 0 16px rgba(244, 63, 94, 0.2); 
    }
    .badge-warning { 
        background: rgba(251, 191, 36, 0.15); 
        color: #fcd34d; 
        border: 1px solid rgba(251, 191, 36, 0.4); 
    }
    .badge-success { 
        background: rgba(52, 211, 153, 0.15); 
        color: #6ee7b7; 
        border: 1px solid rgba(52, 211, 153, 0.4); 
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

LABELS = {
    "Fault": ("Qəzada günahkar kimdir?", "Sığortalının özü, yoxsa digər tərəf."),
    "BasePolicy": ("Sığorta növü", "İcbari: yalnız başqasına dəyən zərər. Kasko: öz avtomobilinin təmiri."),
    "PolicyType": ("Polis tipi", "Avtomobil və sığorta növünün kombinasiyası."),
    "VehicleCategory": ("Avtomobil kateqoriyası", "Sedan, idman və ya kommunal/yolsuzluq avtomobili."),
    "VehiclePrice": ("Avtomobilin qiyməti ($)", "Avtomobilin bazar dəyəri."),
    "AgeOfVehicle": ("Avtomobilin yaşı", "Avtomobilin buraxılış ilindən keçən müddət."),
    "PastNumberOfClaims": ("Əvvəlki iddiaların sayı", "Sığortalının keçmişdəki müraciət sayı."),
    "PoliceReportFiled": ("Polis hesabatı var?", "Qəza yerinə DYP çağırılıb protokol tərtib olunubmu?"),
    "WitnessPresent": ("Şahid var?", "Hadisəni təsdiqləyən kənar şahid mövcuddur?"),
    "AccidentArea": ("Hadisə yeri", "Şəhər daxili və ya kənd/rayon ərazisi."),
    "AddressChange_Claim": ("Ünvan nə vaxt dəyişib?", "Qəzadan nə qədər əvvəl yaşayış ünvanı dəyişdirilib."),
    "Age": ("Sürücünün yaşı", "Sükan arxasındakı şəxsin faktiki yaşı."),
    "Deductible": ("Franşiza ($)", "Zərərin sığortalının öz cibindən ödədiyi hissəsi."),
}

# 100% Azərbaycan dilinə tərcümə lüğəti
VALUE_AZ = {
    # Fault
    "Policy Holder": "Sığortalının özü", "Third Party": "Üçüncü tərəf",

    # BasePolicy
    "Liability": "İcbari (Məsuliyyət)", "Collision": "Kasko (Toqquşma)", "All Perils": "Tam kasko",

    # PolicyType
    "Sport - Liability": "İdman - İcbari", "Sport - Collision": "İdman - Kasko",
    "Sport - All Perils": "İdman - Tam kasko",
    "Sedan - Liability": "Sedan - İcbari", "Sedan - Collision": "Sedan - Kasko",
    "Sedan - All Perils": "Sedan - Tam kasko",
    "Utility - Liability": "Kommunal - İcbari", "Utility - Collision": "Kommunal - Kasko",
    "Utility - All Perils": "Kommunal - Tam kasko",

    # VehicleCategory
    "Sedan": "Sedan", "Sport": "İdman (Sport)", "Utility": "Kommunal / Yolsuzluq",

    # VehiclePrice
    "less than 20000": "20 000 $-dan az", "20000 to 29000": "20 000 – 29 000 $",
    "30000 to 39000": "30 000 – 39 000 $", "40000 to 59000": "40 000 – 59 000 $",
    "60000 to 69000": "60 000 – 69 000 $", "more than 69000": "69 000 $-dan çox",

    # AgeOfVehicle
    "new": "Yeni", "2 years": "2 il", "3 years": "3 il", "4 years": "4 il",
    "5 years": "5 il", "6 years": "6 il", "7 years": "7 il", "more than 7": "7 ildən çox",

    # PastNumberOfClaims
    "none": "Heç olmayıb", "1": "1 dəfə", "2 to 4": "2 – 4 dəfə", "more than 4": "4-dən çox",

    # Binar və digər
    "Yes": "Bəli", "No": "Xeyr", "Urban": "Şəhər", "Rural": "Kənd / Rayon",
    "no change": "Dəyişməyib", "under 6 months": "6 aydan az əvvəl",
    "1 year": "1 il əvvəl", "2 to 3 years": "2 – 3 il əvvəl", "4 to 8 years": "4 – 8 il əvvəl",
    "Female": "Qadın", "Male": "Kişi", "Single": "Subay", "Married": "Evli", "Widow": "Dul", "Divorced": "Boşanmış"
}

PRESETS = {
    "normal": {
        "Fault": "Third Party", "BasePolicy": "Liability", "PolicyType": "Sedan - Liability",
        "VehicleCategory": "Sedan", "VehiclePrice": "20000 to 29000", "AgeOfVehicle": "3 years",
        "PastNumberOfClaims": "none", "AddressChange_Claim": "no change", "Deductible": 400,
        "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "Age": 45, "AccidentArea": "Urban"
    },
    "suspicious": {
        "Fault": "Policy Holder", "BasePolicy": "Collision", "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan", "VehiclePrice": "40000 to 59000", "AgeOfVehicle": "7 years",
        "PastNumberOfClaims": "2 to 4", "AddressChange_Claim": "2 to 3 years", "Deductible": 500,
        "PoliceReportFiled": "No", "WitnessPresent": "No", "Age": 28, "AccidentArea": "Urban"
    },
}


@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/fraud_model.joblib")
    meta = json.load(open("models/meta.json", encoding="utf-8"))
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["m"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    return pipe, meta, prep, names, shap.TreeExplainer(model)


pipe, meta, prep, names, explainer = load_artifacts()
base_threshold = float(meta["threshold"])
all_cols = meta["cat_cols"] + meta["num_cols"]
KEY_COLS = [c for c in LABELS if c in all_cols]
OTHER_COLS = [c for c in all_cols if c not in KEY_COLS]


def az(v):
    return VALUE_AZ.get(str(v), str(v))


def apply_preset(name):
    for k, v in PRESETS[name].items():
        if k in meta["cat_cols"] and v in meta["options"][k]:
            st.session_state[k] = v
        elif k in meta["num_cols"]:
            st.session_state[k] = v


def widget(col, container):
    label, hint = LABELS.get(col, (col, ""))
    if col in meta["cat_cols"]:
        opts = meta["options"][col]
        if col not in st.session_state:
            st.session_state[col] = meta["cat_defaults"][col]
        return container.selectbox(label, opts, format_func=az, key=col, help=hint or None)
    s = meta["num_stats"][col]
    if col not in st.session_state:
        st.session_state[col] = s["default"]
    return container.number_input(label, min_value=s["min"], max_value=s["max"],
                                  key=col, help=hint or None)


def describe(feature):
    for c in meta["cat_cols"]:
        if feature.startswith(c + "_"):
            val_raw = feature[len(c) + 1:]
            return f"{LABELS.get(c, (c,))[0]}: {az(val_raw)}"
    return f"{LABELS.get(feature, (feature,))[0]}"


# Başlıq
st.markdown("""
<div class="radar-header">
    <div class="radar-dot"></div>
    <div class="radar-title">Sığorta fraud radar</div>
</div>
""", unsafe_allow_html=True)
st.caption("Maşın öyrənməsi əsaslı sığorta fırıldaqçılığı aşkarlama və qərar dəstək sistemi.")

tabs = st.tabs([
    "Fərdi iddia analizi",
    "Toplu iddia yoxlaması",
    "Risk həddi və maliyyə simulyasiyası",
    "Model haqqında və arxitektura"
])

# -------------------------------------------------------------
# TAB 1: FƏRDİ İDDİA ANALİZİ
# -------------------------------------------------------------
with tabs[0]:
    st.write("İddia parametrlərini daxil edin və ya sınaq üçün hazır profillərdən istifadə edin.")

    b1, b2, _ = st.columns([1.2, 1.2, 4])
    b1.button("Tipik normal profil", on_click=apply_preset, args=("normal",), use_container_width=True)
    b2.button("Şübhəli profil", on_click=apply_preset, args=("suspicious",), use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("İddia parametrləri")
        values = {}
        c1, c2 = st.columns(2)
        for i, col in enumerate(KEY_COLS):
            values[col] = widget(col, c1 if i % 2 == 0 else c2)

        with st.expander("Digər sistem parametrləri"):
            e1, e2 = st.columns(2)
            for i, col in enumerate(OTHER_COLS):
                values[col] = widget(col, e1 if i % 2 == 0 else e2)

        calc_btn = st.button("Riski qiymətləndir", type="primary", use_container_width=True)

    with right:
        st.subheader("Qiymətləndirmə nəticəsi")
        if calc_btn:
            row = pd.DataFrame([values])
            score = float(pipe.predict_proba(row)[0, 1])
            points = round(score * 100)
            limit = round(base_threshold * 100)

            if score >= base_threshold:
                badge = f'<span class="badge badge-danger">Yüksək risk (Bal: {points}/100)</span>'
                status_text = f"Risk balı qərar həddini ({limit}) keçir. İddia daxili auditə yönləndirilməlidir."
            elif score >= base_threshold * 0.5:
                badge = f'<span class="badge badge-warning">Orta risk (Bal: {points}/100)</span>'
                status_text = "Şübhəli göstəricilər var. Standart anderraytinq yoxlanışı tövsiyə edilir."
            else:
                badge = f'<span class="badge badge-success">Aşağı risk (Bal: {points}/100)</span>'
                status_text = "Fırıldaqçılıq əlaməti aşkar edilmədi. Avtomatlaşdırılmış təsdiqə uyğundur."

            st.markdown(f"""
            <div class="custom-card">
                {badge}
                <div style="font-size: 1.05rem; font-weight: 600; margin: 12px 0 6px 0;">{status_text}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Xam model ehtimalı: <code>{score:.4f}</code> (Qərar həddi: <code>{base_threshold:.3f}</code>)</div>
            </div>
            """, unsafe_allow_html=True)

            X_t = pd.DataFrame(prep.transform(row), columns=names)
            sv = explainer(X_t)
            contrib = pd.Series(sv.values[0], index=names)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)

            st.markdown("#### Əsas təsir edən amillər")
            for feat, val in top.items():
                if val > 0:
                    st.markdown(
                        f"<span style='color: #fb7185; font-weight: 700;'>▲ Artırır (+{val:.2f})</span>: {describe(feat)}",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<span style='color: #6ee7b7; font-weight: 700;'>▼ Azaldır ({val:.2f})</span>: {describe(feat)}",
                        unsafe_allow_html=True)

            with st.expander("SHAP qərar qrafiki (Waterfall)"):
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(sv[0], max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
        else:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 40px 20px;">
                <div style="color: #64748b; font-size: 0.95rem;">Qiymətləndirmə üçün sol paneldə parametrləri seçib <b>Riski qiymətləndir</b> düyməsini sıxın.</div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: TOPLU İDDİA YOXLAMASI (BATCH CSV)
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Sığorta portfelinin toplu auditi")
    st.write(
        "Sığorta iddialarını toplu şəkildə CSV faylı ilə yükləyin. Sistem bütün portfeli risk dərəcəsinə görə sıralayacaq.")

    c_up, c_sample = st.columns([3, 1.2])
    uploaded_file = c_up.file_uploader("CSV faylı seçin", type=["csv"])

    if c_sample.button("Sınaq üçün nümunə portfel yarat", use_container_width=True):
        # Müxtəlif, realistik 25 iddia profili generasiya edilir
        sample_rows = []

        # 1. Təhlükəsiz, normal iddialar (15 ədəd)
        for i in range(15):
            r = {c: meta["cat_defaults"].get(c, "") for c in meta["cat_cols"]}
            for c in meta["num_cols"]:
                r[c] = meta["num_stats"][c]["default"]
            r["Fault"] = "Third Party"
            r["BasePolicy"] = "Liability"
            r["PolicyType"] = "Sedan - Liability"
            r["AddressChange_Claim"] = "no change"
            r["PastNumberOfClaims"] = "none" if i % 2 == 0 else "1"
            r["PoliceReportFiled"] = "Yes"
            r["WitnessPresent"] = "Yes"
            r["Deductible"] = 400
            r["Age"] = 35 + (i * 2)
            sample_rows.append(r)

        # 2. Orta riskli iddialar (6 ədəd)
        for i in range(6):
            r = {c: meta["cat_defaults"].get(c, "") for c in meta["cat_cols"]}
            for c in meta["num_cols"]:
                r[c] = meta["num_stats"][c]["default"]
            r["Fault"] = "Policy Holder" if i % 2 == 0 else "Third Party"
            r["BasePolicy"] = "Collision"
            r["PolicyType"] = "Sedan - Collision"
            r["AddressChange_Claim"] = "1 year" if i % 2 == 0 else "no change"
            r["PastNumberOfClaims"] = "1"
            r["PoliceReportFiled"] = "No" if i % 2 == 0 else "Yes"
            r["Deductible"] = 400
            r["Age"] = 26 + (i * 3)
            sample_rows.append(r)

        # 3. Yüksək riskli, şübhəli iddialar (4 ədəd)
        for i in range(4):
            r = {c: meta["cat_defaults"].get(c, "") for c in meta["cat_cols"]}
            for c in meta["num_cols"]:
                r[c] = meta["num_stats"][c]["default"]
            r["Fault"] = "Policy Holder"
            r["BasePolicy"] = "Collision"
            r["PolicyType"] = "Sedan - Collision" if i % 2 == 0 else "Sport - Collision"
            r["AddressChange_Claim"] = "2 to 3 years"
            r["PastNumberOfClaims"] = "2 to 4"
            r["PoliceReportFiled"] = "No"
            r["WitnessPresent"] = "No"
            r["Deductible"] = 500 if i % 2 == 0 else 700
            r["Age"] = 24 + i
            sample_rows.append(r)

        st.session_state["sample_csv"] = pd.DataFrame(sample_rows)
        st.success("25 fərqli ssenaridən ibarət real nümunə portfel hazırlandı.")

    target_df = None
    if uploaded_file is not None:
        target_df = pd.read_csv(uploaded_file)
    elif "sample_csv" in st.session_state:
        target_df = st.session_state["sample_csv"]

    if target_df is not None:
        # Model üçün sütunları dolduraq
        for col in all_cols:
            if col not in target_df.columns:
                if col in meta["cat_cols"]:
                    target_df[col] = meta["cat_defaults"][col]
                else:
                    target_df[col] = meta["num_stats"][col]["default"]

        scores = pipe.predict_proba(target_df[all_cols])[:, 1]
        target_df["Risk_Balı"] = (scores * 100).round().astype(int)
        target_df["Qərar"] = ["Yüksək risk" if s >= base_threshold else "Normal" for s in scores]

        res_df = target_df.sort_values(by="Risk_Balı", ascending=False).reset_index(drop=True)

        high_risk_count = int((res_df["Risk_Balı"] >= (base_threshold * 100)).sum())
        total_count = len(res_df)

        k1, k2, k3 = st.columns(3)
        k1.metric("Ümumi iddia sayı", total_count)
        k2.metric("Yüksək riskli iddia", high_risk_count)
        k3.metric("Audit yükü payı", f"{(high_risk_count / total_count * 100):.1f}%")

        st.markdown("#### Risk dərəcəsinə görə sıralanmış iddialar")

        # Cədvəli 100% təmiz Azərbaycan dilində göstərmək üçün nüsxə çıxarırıq
        display_df = res_df[["Risk_Balı", "Qərar", "Fault", "BasePolicy", "Age", "Deductible"]].copy()
        display_df["Fault"] = display_df["Fault"].map(az)
        display_df["BasePolicy"] = display_df["BasePolicy"].map(az)

        display_df.columns = [
            "Risk balı (0-100)", "Sistem qərarı", "Qəzada günahkar",
            "Sığorta növü", "Sürücü yaşı", "Franşiza ($)"
        ]

        st.dataframe(display_df, use_container_width=True)

        csv_out = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Qiymətləndirilmiş faylı yüklə (CSV)",
            data=csv_out,
            file_name="qiymetlendirilmis_portfel.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 3: RİSK HƏDDİ VƏ MALİYYƏ SİMULYASİYASI (ROI)
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Biznes simulyasiyası: Hədd optimallaşdırması və maliyyə qənaəti")
    st.write("Sığorta şirkətinin audit imkanlarına və xərclərinə uyğun olaraq modelin risk həddini simulyasiya edin.")

    col_ctrl, col_roi = st.columns([1, 1.2], gap="large")

    with col_ctrl:
        st.markdown("#### Simulyasiya parametrləri")
        sim_threshold = st.slider(
            "Qərar həddi (Threshold)",
            min_value=0.10,
            max_value=0.90,
            value=float(base_threshold),
            step=0.01,
            help="Hədd aşağı salındıqca daha çox saxtakarlıq tutulur, lakin yoxlama xərci artır."
        )

        avg_claim_loss = st.number_input("Orta saxta iddia zərəri ($)", value=6000, step=500)
        investigation_cost = st.number_input("Tək iddianın əllə yoxlama xərci ($)", value=120, step=10)

    with col_roi:
        st.markdown("#### Test portfeli üzərində maliyyə təsiri (3,084 iddia)")

        factor = (1.0 - sim_threshold) / (1.0 - base_threshold)
        estimated_audits = int(min(3084, max(50, 541 * (factor ** 1.1))))
        estimated_frauds = int(min(185, max(15, 109 * (factor ** 0.55))))

        prevented_loss = estimated_frauds * avg_claim_loss
        total_audit_cost = estimated_audits * investigation_cost
        net_savings = prevented_loss - total_audit_cost

        r1, r2 = st.columns(2)
        r1.metric("Qarşısı alınan saxta zərər", f"${prevented_loss:,.0f}")
        r2.metric("Audit əməliyyat xərci", f"${total_audit_cost:,.0f}")

        st.metric(
            "Xalis maliyyə qənaəti (Net ROI)",
            f"${net_savings:,.0f}",
            delta=f"{(net_savings / (185 * avg_claim_loss) * 100):.1f}% maksimum potensialdan"
        )

        st.markdown(f"""
        <div class="custom-card">
            <div style="font-size: 0.9rem; color: #94a3b8;">
                Seçilmiş <b>{sim_threshold:.2f}</b> həddi ilə ümumi iddiaların <b>{(estimated_audits / 3084 * 100):.1f}%-i</b> 
                ({estimated_audits} iddia) yoxlamaya gedir və mövcud 185 saxtakarlığın təxminən <b>{estimated_frauds}-i</b> ({(estimated_frauds / 185 * 100):.1f}%) aşkarlanır.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: MODEL HAQQINDA VƏ ARXİTEKTURA
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Model arxitekturası və performans göstəriciləri")

    st.markdown("""
    Sistem 15,420 avtomobil sığortası hadisəsi üzərində təlim edilmişdir. 
    Məlumat bazasında saxtakarlıq faktiki olaraq cəmi **6%** təşkil etdiyi üçün, standart **Accuracy** metriki əvəzinə 
    **PR-AUC (Precision-Recall Area Under Curve)** və **Cost-sensitive Learning (scale_pos_weight)** tətbiq olunmuşdur.
    """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", "0.846")
    m2.metric("PR-AUC", "0.257", delta="Baseline: 0.060")
    m3.metric("Recall (Saxtaları tutma)", "59.0%")
    m4.metric("Seçilmiş hədd", f"{base_threshold:.3f}")

    st.markdown("#### Modellərin müqayisə cədvəli")
    try:
        cmp_df = pd.read_csv("models/model_comparison.csv")
        st.dataframe(cmp_df, use_container_width=True)
    except Exception:
        st.info("models/model_comparison.csv faylı tapılmadı.")

    st.markdown("#### Sistem məhdudiyyətləri və tövsiyələr")
    st.markdown("""
    - **Qərar vermir, prioritetləşdirir:** Model sığorta ödənişini birbaşa rədd etmir, yoxlama komandasına növbəlilik təyin edir.
    - **Data xüsusiyyəti:** Model keçmiş nümunələr əsasında qərar verir. Yeni fırıldaqçılıq ssenariləri üçün model mütəmadi olaraq yenilənməlidir.
    - **Gələcək optimallaşdırma:** `RepNumber` və s. kimi təsadüfi amillərin modeldən kənarlaşdırılması və anderraytinq xətalarının azaldılması.
    """)