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

# Enterprise Dark Minimalist Stil
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    h1, h2, h3, h4 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #f8fafc !important;
    }

    /* Metrik kartları */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }

    /* Tablar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        padding: 0 16px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        background-color: transparent;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }

    /* Kart qutuları */
    .custom-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-danger { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-warning { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .badge-success { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }

    /* Düymələr */
    .stButton>button {
        font-weight: 600;
        border-radius: 6px;
        letter-spacing: -0.01em;
    }
</style>
""", unsafe_allow_html=True)

LABELS = {
    "Fault": ("Qəzada günahkar kimdir?", "Sığortalının özü, yoxsa digər tərəf."),
    "BasePolicy": ("Sığorta növü",
                   "Liability: yalnız başqasına dəyən ziyan. Collision: öz avtomobilinin zədəsi. All Perils: hər ikisi."),
    "PolicyType": ("Polis tipi", "Avtomobil kateqoriyası və sığorta növünün birləşməsi."),
    "VehicleCategory": ("Avtomobil kateqoriyası", "Sedan, idman və ya kommunal avtomobil."),
    "VehiclePrice": ("Avtomobilin qiyməti ($)", "Qiymət aralığı."),
    "AgeOfVehicle": ("Avtomobilin yaşı", "Neçə illik avtomobildir."),
    "PastNumberOfClaims": ("Əvvəlki iddiaların sayı", "Sığortalı bundan əvvəl neçə iddia verib."),
    "PoliceReportFiled": ("Polis hesabatı var?", "Hadisə polisə bildirilib?"),
    "WitnessPresent": ("Şahid var?", "Hadisənin şahidi olub?"),
    "AccidentArea": ("Hadisə yeri", "Şəhər və ya kənd."),
    "AddressChange_Claim": ("Ünvan nə vaxt dəyişib?", "İddiadan nə qədər əvvəl sığortalı ünvanını dəyişib."),
    "Age": ("Sürücünün yaşı", ""),
    "Deductible": ("Franşiza ($)", "Zərərin hansı hissəsini sığortalı özü ödəyir."),
}

VALUE_AZ = {
    "Policy Holder": "Sığortalının özü", "Third Party": "Üçüncü tərəf",
    "Liability": "Liability (məsuliyyət)", "Collision": "Collision (toqquşma)",
    "Yes": "Bəli", "No": "Xeyr", "Urban": "Şəhər", "Rural": "Kənd",
    "no change": "Dəyişməyib", "under 6 months": "6 aydan az",
    "1 year": "1 il", "2 to 3 years": "2-3 il", "4 to 8 years": "4-8 il",
    "none": "Heç olmayıb", "more than 4": "4-dən çox",
}

PRESETS = {
    "normal": {
        "Fault": "Third Party", "BasePolicy": "Liability", "PolicyType": "Sedan - Liability",
        "AddressChange_Claim": "no change", "Deductible": 400,
        "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "Age": 45
    },
    "suspicious": {
        "Fault": "Policy Holder", "BasePolicy": "Collision", "PolicyType": "Sedan - Collision",
        "AddressChange_Claim": "2 to 3 years", "Deductible": 500,
        "PoliceReportFiled": "No", "WitnessPresent": "No", "Age": 29
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
            return f"{LABELS.get(c, (c,))[0]}: {az(feature[len(c) + 1:])}"
    return f"{LABELS.get(feature, (feature,))[0]}"


# Başlıq
st.title("Sığorta fraud radar")
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

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

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

            # Risk statusu
            if score >= base_threshold:
                badge = f'<span class="badge badge-danger">Yüksək risk (Bal: {points}/100)</span>'
                status_text = f"Risk balı qərar həddini ({limit}) keçir. İddia dərhal daxili auditə yönləndirilməlidir."
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

            # SHAP İzahı
            X_t = pd.DataFrame(prep.transform(row), columns=names)
            sv = explainer(X_t)
            contrib = pd.Series(sv.values[0], index=names)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)

            st.markdown("#### Əsas təsir edən amillər")
            for feat, val in top.items():
                if val > 0:
                    st.markdown(
                        f"<span style='color: #f43f5e; font-weight: 700;'>▲ Artırır (+{val:.2f})</span>: {describe(feat)}",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<span style='color: #34d399; font-weight: 700;'>▼ Azaldır ({val:.2f})</span>: {describe(feat)}",
                        unsafe_allow_html=True)

            with st.expander("SHAP qərar qrafiki (Waterfall)"):
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(sv[0], max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
        else:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 40px 20px;">
                <div style="color: #64748b; font-size: 0.95rem;">Qiymətləndirmə üçün sol paneldə məlumatları daxil edib <b>Riski qiymətləndir</b> düyməsini sıxın.</div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: TOPLU İDDİA YOXLAMASI (BATCH CSV)
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Sığorta portfelinin toplu auditi")
    st.write(
        "Sığorta iddialarını toplu şəkildə CSV faylı ilə yükləyin. Sistem bütün portfeli prioritet sırasına düzür.")

    # Nümunə yaratmaq imkanı
    c_up, c_sample = st.columns([3, 1])
    uploaded_file = c_up.file_uploader("CSV faylı seçin", type=["csv"])

    if c_sample.button("Sınaq üçün nümunə generasiya et"):
        # Sessiyada saxta test datası yaradılır
        sample_rows = []
        for _ in range(25):
            r = {c: meta["cat_defaults"].get(c, "") for c in meta["cat_cols"]}
            for c in meta["num_cols"]:
                r[c] = meta["num_stats"][c]["default"]
            sample_rows.append(r)
        # Bəzilərini şübhəli edək
        sample_rows[0]["Fault"] = "Policy Holder"
        sample_rows[0]["AddressChange_Claim"] = "2 to 3 years"
        sample_rows[1]["Fault"] = "Policy Holder"
        sample_rows[1]["Deductible"] = 700

        sample_df = pd.DataFrame(sample_rows)
        csv_data = sample_df.to_csv(index=False).encode('utf-8')
        st.session_state["sample_csv"] = sample_df
        st.success("25 sətirlik nümunə portfel hazırlandı. Aşağıdakı cədvələ baxın.")

    target_df = None
    if uploaded_file is not None:
        target_df = pd.read_csv(uploaded_file)
    elif "sample_csv" in st.session_state:
        target_df = st.session_state["sample_csv"]

    if target_df is not None:
        # Uyğun sütunları dolduraq
        for col in all_cols:
            if col not in target_df.columns:
                if col in meta["cat_cols"]:
                    target_df[col] = meta["cat_defaults"][col]
                else:
                    target_df[col] = meta["num_stats"][col]["default"]

        # Skorinq
        scores = pipe.predict_proba(target_df[all_cols])[:, 1]
        target_df["Risk_Balı"] = (scores * 100).round().astype(int)
        target_df["Qərar"] = ["Yüksək risk" if s >= base_threshold else "Normal" for s in scores]

        # Sıralama
        res_df = target_df.sort_values(by="Risk_Balı", ascending=False)

        high_risk_count = (res_df["Risk_Balı"] >= (base_threshold * 100)).sum()
        total_count = len(res_df)

        k1, k2, k3 = st.columns(3)
        k1.metric("Ümumi iddia sayı", total_count)
        k2.metric("Yüksək riskli iddia", high_risk_count)
        k3.metric("Yoxlama yükü payı", f"{(high_risk_count / total_count * 100):.1f}%")

        st.markdown("#### Risk dərəcəsinə görə sıralanmış iddialar")
        cols_to_show = ["Risk_Balı", "Qərar", "Fault", "BasePolicy", "Age", "Deductible"]
        present_cols = [c for c in cols_to_show if c in res_df.columns]
        st.dataframe(res_df[present_cols], use_container_width=True)

        # Nəticəni yükləmə düyməsi
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

        # Dinamik hesablama (təxmini test portfeli paylanmasına əsasən)
        # Model 0.604 həddində: 541 yoxlanış, 109 tapılan saxta
        # Hədd dəyişdikdə nisbət tənzimlənir
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
    - **Gələcək optimallaşdırma:** `RepNumber` və s. kimi təsadüfi identifikatorların modeldən kənarlaşdırılması və anderraytinq xətalarının azaldılması.
    """)