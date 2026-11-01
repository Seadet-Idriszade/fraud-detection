import json
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sığorta Fraud Radar", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/fraud_model.joblib")
    meta = json.load(open("models/meta.json", encoding="utf-8"))
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["m"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    explainer = shap.TreeExplainer(model)
    return pipe, meta, prep, names, explainer


pipe, meta, prep, names, explainer = load_artifacts()
threshold = meta["threshold"]
all_cols = meta["cat_cols"] + meta["num_cols"]

KEY_COLS = ["Fault", "BasePolicy", "PolicyType", "VehicleCategory", "VehiclePrice",
            "AgeOfVehicle", "PastNumberOfClaims", "PoliceReportFiled", "WitnessPresent",
            "AccidentArea", "AddressChange_Claim", "Age", "Deductible"]
KEY_COLS = [c for c in KEY_COLS if c in all_cols]
OTHER_COLS = [c for c in all_cols if c not in KEY_COLS]


def widget(col, container):
    if col in meta["cat_cols"]:
        opts = meta["options"][col]
        return container.selectbox(col, opts, index=opts.index(meta["cat_defaults"][col]), key=col)
    s = meta["num_stats"][col]
    return container.number_input(col, min_value=s["min"], max_value=s["max"],
                                  value=s["default"], key=col)


st.title("🛡️ Sığorta Fraud Radar")
st.caption("Sığorta iddiasının şübhəli olma riskini hesablayan və səbəbini izah edən AI sistemi")

tab_demo, tab_about = st.tabs(["🔍 İddianı yoxla", "📊 Model haqqında"])

with tab_demo:
    left, right = st.columns(2)

    with left:
        st.subheader("İddia məlumatları")
        values = {}
        c1, c2 = st.columns(2)
        for i, col in enumerate(KEY_COLS):
            values[col] = widget(col, c1 if i % 2 == 0 else c2)
        with st.expander("Digər sahələr (standart dəyərlərlə doldurulub)"):
            e1, e2 = st.columns(2)
            for i, col in enumerate(OTHER_COLS):
                values[col] = widget(col, e1 if i % 2 == 0 else e2)
        run = st.button("Riski hesabla", type="primary")

    with right:
        st.subheader("Nəticə")
        if run:
            row = pd.DataFrame([values])
            score = float(pipe.predict_proba(row)[0, 1])

            st.metric("Risk skoru", f"{score:.1%}")
            st.progress(min(score, 1.0))
            if score >= threshold:
                st.error("🔴 YÜKSƏK RİSK: iddia əl ilə yoxlanmalıdır")
            elif score >= threshold * 0.5:
                st.warning("🟡 ORTA RİSK: diqqətlə baxın")
            else:
                st.success("🟢 AŞAĞI RİSK")
            st.caption(f"Qərar həddi: {threshold:.2f}. Bu, dəqiq ehtimal yox, risk skorudur.")

            st.markdown("**Niyə belə qiymətləndirildi?**")
            X_t = pd.DataFrame(prep.transform(row), columns=names)
            sv = explainer(X_t)
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

            top = pd.Series(sv.values[0], index=names).sort_values(key=abs, ascending=False).head(4)
            for n, v in top.items():
                st.write(("🔺 Riski artırır: " if v > 0 else "🔻 Riski azaldır: ") + f"**{n}**")
        else:
            st.info("Sol tərəfdə məlumatları seçin və düyməyə basın.")

with tab_about:
    st.subheader("Model müqayisəsi")
    st.dataframe(pd.read_csv("models/model_comparison.csv"))
    st.markdown("""
**Problem:** Data-da fraud təxminən 6%-dir. Hər şeyə "normal" deyən model 94% dəqiqlik alır, amma heç bir fraud tapmır.
Buna görə dəqiqlik (accuracy) yox, **PR-AUC**, **recall** və **precision** ilə qiymətləndirdik.

**Həll:** class weights, SMOTE və XGBoost-un `scale_pos_weight` üsullarını müqayisə etdik.

**İzah:** SHAP hər iddia üçün hansı amilin riski artırdığını və ya azaltdığını göstərir.

**Məhdudiyyət:** Model yalnız tarixi datadakı nümunələri öyrənir, nəticə qərarı verən yox, yoxlama komandasına kömək edən siqnaldır.
""")