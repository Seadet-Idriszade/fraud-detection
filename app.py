import json
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sığorta Fraud Radar", page_icon="🛡️", layout="wide")

LABELS = {
    "Fault": ("Qəzada günahkar kimdir?", "Sığortalının özü, yoxsa digər tərəf."),
    "BasePolicy": ("Sığorta növü", "Liability: yalnız başqasına dəyən ziyan. Collision: öz avtomobilinin zədəsi. All Perils: hər ikisi."),
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
    "normal": {"Fault": "Third Party", "BasePolicy": "Liability", "PolicyType": "Sedan - Liability",
               "AddressChange_Claim": "no change", "Deductible": 400,
               "PoliceReportFiled": "Yes", "WitnessPresent": "Yes"},
    "suspicious": {"Fault": "Policy Holder", "BasePolicy": "Collision", "PolicyType": "Sedan - Collision",
                   "AddressChange_Claim": "2 to 3 years", "Deductible": 500,
                   "PoliceReportFiled": "No", "WitnessPresent": "No"},
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
threshold = meta["threshold"]
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


def describe(feature, value):
    for c in meta["cat_cols"]:
        if feature.startswith(c + "_"):
            return f"{LABELS.get(c, (c,))[0]}: {az(feature[len(c) + 1:])}"
    return f"{LABELS.get(feature, (feature,))[0]}"


st.title("🛡️ Sığorta Fraud Radar")
st.caption("Sığorta iddiasının saxta ola biləcəyini qiymətləndirən və səbəbini izah edən AI sistemi. "
           "Sistem qərar vermir, yoxlama əməkdaşına hansı iddiaya əvvəl baxmağı göstərir.")

tab_demo, tab_about = st.tabs(["🔍 İddianı yoxla", "📊 Model haqqında"])

with tab_demo:
    st.write("**Sürətli sınaq:** hazır nümunələrdən birini seçin və ya məlumatları özünüz dəyişin.")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🟢 Normal iddia nümunəsi", on_click=apply_preset, args=("normal",))
    b2.button("🔴 Şübhəli iddia nümunəsi", on_click=apply_preset, args=("suspicious",))

    left, right = st.columns(2)
    with left:
        st.subheader("İddia məlumatları")
        values = {}
        c1, c2 = st.columns(2)
        for i, col in enumerate(KEY_COLS):
            values[col] = widget(col, c1 if i % 2 == 0 else c2)
        with st.expander("Digər sahələr (adətən ən çox rast gəlinən dəyərlə doldurulub)"):
            e1, e2 = st.columns(2)
            for i, col in enumerate(OTHER_COLS):
                values[col] = widget(col, e1 if i % 2 == 0 else e2)
        run = st.button("Riski hesabla", type="primary")

    with right:
        st.subheader("Nəticə")
        if run:
            row = pd.DataFrame([values])
            score = float(pipe.predict_proba(row)[0, 1])
            points = round(score * 100)
            limit = round(threshold * 100)

            st.metric("Risk balı (0-100)", f"{points}")
            st.progress(min(score, 1.0))
            if score >= threshold:
                st.error(f"🔴 YÜKSƏK RİSK. Bal {limit}-dən yuxarıdır, iddia əl ilə yoxlanmalıdır.")
            elif score >= threshold * 0.5:
                st.warning("🟡 ORTA RİSK. Diqqətlə baxmaq tövsiyə olunur.")
            else:
                st.success("🟢 AŞAĞI RİSK. Əlavə yoxlama tələb olunmur.")
            st.caption("Bu bal ehtimal faizi deyil, iddiaları bir-biri ilə müqayisə etmək üçün göstəricidir.")

            X_t = pd.DataFrame(prep.transform(row), columns=names)
            sv = explainer(X_t)
            contrib = pd.Series(sv.values[0], index=names)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)

            st.markdown("**Bu nəticəyə ən çox nə təsir etdi?**")
            for feat, val in top.items():
                arrow = "🔺 riski artırır" if val > 0 else "🔻 riski azaldır"
                st.write(f"{arrow}: **{describe(feat, val)}**")

            with st.expander("Texniki izah (SHAP qrafiki)"):
                shap.plots.waterfall(sv[0], max_display=10, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
        else:
            st.info("Soldakı düymələrdən nümunə seçin və ya məlumatları dəyişin, sonra "
                    "**Riski hesabla** düyməsinə basın.")

with tab_about:
    st.subheader("Bu sistem nə edir?")
    st.write("Sığorta şirkətinə gündə çoxlu iddia gəlir və hamısını əllə yoxlamaq mümkün deyil. "
             "Sistem 15 mindən çox keçmiş iddiadan öyrənib və yeni iddiaya risk balı verir.")
    st.subheader("Əsas çətinlik")
    st.write("İddiaların yalnız təxminən 6%-i saxtadır. Hər iddiaya 'normal' deyən model 94% dəqiq olardı, "
             "amma heç bir saxta iddia tapmazdı. Buna görə dəqiqliyə yox, saxta iddiaları tapma keyfiyyətinə baxdıq.")
    st.subheader("Nəticə (modelin görmədiyi 3084 iddia üzərində)")
    st.write("- 185 saxta iddianın **109-u (59%)** tapıldı, 76-sı qaçırıldı.\n"
             "- Sistem 541 iddianı şübhəli saydı, onların təxminən **hər 5-dən 1-i** həqiqətən saxta idi.\n"
             "- Əgər yoxlama komandası təsadüfi iddiaları seçsəydi, saxtaların 59%-ni tutmaq üçün iddiaların "
             "təxminən 59%-nə baxmalı olardı. Sistemlə isə 17,5%-ə baxmaq kifayət edir.")
    st.subheader("Modellərin müqayisəsi")
    st.dataframe(pd.read_csv("models/model_comparison.csv"))
    st.subheader("Məhdudiyyətlər")
    st.write("Model yalnız keçmiş datadakı nümunələri öyrənir, yeni növ saxtakarlığı tanımaya bilər. "
             "Şübhəli iddiaların çoxu əslində normaldır, ona görə sistem qərar verməməli, yalnız prioritet göstərməlidir.")