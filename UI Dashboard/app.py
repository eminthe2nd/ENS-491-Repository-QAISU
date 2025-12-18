# app.py
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import altair as alt

from engine import QualityDecisionSystem


# -----------------------------
# SAME CONFIG AS NOTEBOOK
# -----------------------------
EXAMPLE_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "low_confidence_threshold": 0.6,
    "high_risk_confidence_threshold": 0.4,
    "inconsistency_ratio_threshold": 1.2,
    "inspection_type_weights": {"Gorsel": 1.0, "Fonksiyonel": 1.5, "Tahribatli": 2.0},
    "cost_center_capacity": {"MM01": 1000, "MM02": 800, "MM03": 1200},
    "overload_penalty": 2.0,
    "inspection_workload_hours": {"Gorsel": 0.5, "Fonksiyonel": 1.5, "Tahribatli": 3.0},
    "w_transition": 1.0,
    "w_flow": 1.5,
    "w_anomaly_penalty": 2.0,
    "w_inefficiency_penalty": 1.0,
    "inspection_unit_cost": 1.0,
    "defect_financial_weight": 10.0,
    "handling_costs": {"KABUL": 0.0, "RTV": 1.0, "RU": 2.0, "ISLAH": 5.0, "HURDA": 10.0},
    "cost_vector": [1.0, 5.0, 10.0, 15.0, 20.0],
    "quality_vector": [1.0, 0.8, 0.6, 0.3, 0.9],
    "capacity_vector": [1.0, 0.9, 0.7, 0.5, 0.8],
    "W_COPQ": 1.5,
    "W_CC": 1.0,
    "W_FLOW": 1.5,
    "W_ASSIGN": 2.0,
}

st.set_page_config(page_title="Roketsan QAISU Workbench", page_icon="🛰️", layout="wide")


# -----------------------------
# Session state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "📁 Data & Exploration"  # default landing
if "engine" not in st.session_state:
    st.session_state.engine = None
if "df" not in st.session_state:
    st.session_state.df = None
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "feature_eng_done" not in st.session_state:
    st.session_state.feature_eng_done = False


# -----------------------------
# Helpers
# -----------------------------
def safe_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit/pyarrow bazen object kolonlarda (list/dict/mixed) patlıyor.
    Güvenli render için:
      - list/dict gibi tipleri string'e çevir
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(lambda x: str(x) if isinstance(x, (list, dict, set, tuple, np.ndarray)) else x)
    return out


def run_feature_engineering(engine: QualityDecisionSystem):
    # Notebook ile aynı feature engineering zinciri
    engine.perform_eda()
    engine.add_cost_center_features()
    engine.add_transition_features()
    engine.add_inefficiency_features()
    engine.add_pca_features()
    engine.add_mutual_info()
    engine.add_clusters()
    engine.add_anomaly_score()
    engine.add_corr_feature()
    return engine


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss_pct = (miss / max(1, len(df)) * 100).round(2)
    out = pd.DataFrame({"column": miss.index, "missing_count": miss.values, "missing_pct": miss_pct.values})
    out = out.sort_values("missing_count", ascending=False)
    return out


def bar_counts_chart(df: pd.DataFrame, col: str, top_n: int = 15):
    vc = df[col].astype(str).value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]
    chart = (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{col}:N", sort="-x", title=col),
            color=alt.Color(f"{col}:N", legend=None),
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("count:Q")],
        )
        .properties(height=min(450, 22 * len(vc) + 80))
    )
    return chart


def hist_chart(df: pd.DataFrame, col: str, bins: int = 30):
    d = pd.to_numeric(df[col], errors="coerce").dropna()
    if d.empty:
        return None
    tmp = pd.DataFrame({col: d})
    chart = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=f"{col} (binned)"),
            y=alt.Y("count():Q", title="Count of Records"),
            tooltip=[alt.Tooltip("count():Q")],
        )
        .properties(height=260)
    )
    return chart


def boxplot_chart(df: pd.DataFrame, col: str):
    d = pd.to_numeric(df[col], errors="coerce").dropna()
    if d.empty:
        return None
    tmp = pd.DataFrame({col: d, "_": " "})
    chart = (
        alt.Chart(tmp)
        .mark_boxplot()
        .encode(
            x=alt.X("_:N", title=""),
            y=alt.Y(f"{col}:Q", title=col),
            tooltip=[alt.Tooltip(f"{col}:Q")],
        )
        .properties(height=260)
    )
    return chart


def plot_confusion_matrix(cm: np.ndarray, labels):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def feature_importance_chart(engine: QualityDecisionSystem, top_n: int = 20):
    fi = engine.feature_importance
    if fi is None or fi.empty:
        return None
    value_col = "avg_importance" if "avg_importance" in fi.columns else (
        "rf_importance" if "rf_importance" in fi.columns else None
    )
    if value_col is None:
        return None

    top = fi.sort_values(value_col, ascending=False).head(top_n).copy()
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("feature:N", legend=None),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip(f"{value_col}:Q")],
        )
        .properties(height=min(520, 22 * len(top) + 100))
    )
    return chart


# -----------------------------
# Sidebar (3 pages only)
# -----------------------------
st.sidebar.title("🛰️ QAISU Workbench")
st.sidebar.markdown("### Navigation")

if st.sidebar.button("📁 Data & Exploration", use_container_width=True):
    st.session_state.page = "📁 Data & Exploration"
if st.sidebar.button("🧠 Modeling Studio", use_container_width=True):
    st.session_state.page = "🧠 Modeling Studio"
if st.sidebar.button("📊 Decisions & Export", use_container_width=True):
    st.session_state.page = "📊 Decisions & Export"

page = st.session_state.page


# -----------------------------
# PAGE 1: DATA & EXPLORATION (Landing)
# -----------------------------
if page == "📁 Data & Exploration":
    st.title("📁 Data Upload & Exploration")

    uploaded_file = st.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx", "csv"])
    if uploaded_file is not None:
        suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        engine = QualityDecisionSystem(temp_path, config=EXAMPLE_CONFIG)
        engine.load_and_prep_data()

        st.session_state.engine = engine
        st.session_state.df = engine.df_raw
        st.session_state.models_trained = False
        st.session_state.results_df = None
        st.session_state.feature_eng_done = False

    df = st.session_state.df
    engine: Optional[QualityDecisionSystem] = st.session_state.engine

    if df is None:
        st.info("Upload a dataset to start.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{len(df):,}")
        with c2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with c3:
            st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        st.subheader("Preview")
        st.dataframe(safe_df_for_streamlit(df.head(50)), use_container_width=True)

        with st.expander("Show column types"):
            st.write(df.dtypes)

        st.divider()

        st.subheader("Quick charts (common fields)")
        quick_cols = []
        for candidate in ["ACTUAL_DECISION", "DECISION_TYPE", "DEFECT_TYPE", "RESPONSIBLE_UNIT", "PROCESS_TYPE", "MUAYENE_TIPI", "STOCK_PLACE"]:
            if candidate in df.columns:
                quick_cols.append(candidate)

        if quick_cols:
            qc = st.selectbox("Select a quick column", quick_cols, index=0)
            topn = st.slider("Top-N categories", 5, 30, 15)
            st.altair_chart(bar_counts_chart(df, qc, top_n=topn), use_container_width=True)
        else:
            st.info("No columns found for quick chart (ACTUAL_DECISION/DEFECT_TYPE etc.) in this dataset.")

        st.divider()

        st.subheader("Explore any column")
        col = st.selectbox("Column", list(df.columns), index=0)

        is_num = pd.api.types.is_numeric_dtype(df[col])
        uniq = df[col].nunique(dropna=True)

        if is_num:
            bins = st.slider("Histogram bins", 10, 80, 30)
            h = hist_chart(df, col, bins=bins)
            b = boxplot_chart(df, col)
            if h is None or b is None:
                st.warning("Bu numeric kolonda çizim için yeterli veri yok (NaN olabilir).")
            else:
                st.altair_chart(h, use_container_width=True)
                st.altair_chart(b, use_container_width=True)
        else:
            topn = st.slider("Top-N categories (for bar chart)", 5, 50, 20)
            if uniq > 200:
                st.info(f"Bu kolon çok fazla unique değer içeriyor ({uniq}). Top-{topn} gösteriliyor.")
            st.altair_chart(bar_counts_chart(df, col, top_n=topn), use_container_width=True)

        with st.expander("Missing values (top 20)"):
            miss_df = _missing_summary(df).head(20)
            st.dataframe(miss_df, use_container_width=True)


# -----------------------------
# PAGE 2: MODELING STUDIO
# -----------------------------
elif page == "🧠 Modeling Studio":
    st.title("🧠 Modeling Studio")

    engine: Optional[QualityDecisionSystem] = st.session_state.engine
    df = st.session_state.df

    if engine is None or df is None:
        st.warning("Upload a dataset first from Data & Exploration page.")
    

        model_options = ["Random Forest", "XGBoost", "Logistic Regression"]
        selected_models = st.multiselect("Select models to train", model_options, default=model_options)

        colA, colB, colC = st.columns([1.2, 1.2, 1.6])

        with colA:
            rf_estimators = st.slider("RF n_estimators", 50, 500, 200, step=25)
            rf_depth = st.slider("RF max_depth", 3, 30, 15)

        with colB:
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)


        train_clicked = st.button("🚀 Train & Evaluate", type="primary")

        if train_clicked:
            with st.spinner("Running preprocessing + feature engineering + training..."):
                run_feature_engineering(engine)
                st.session_state.feature_eng_done = True

                engine.prepare_for_modeling(test_size=float(test_size))

                if "Random Forest" in selected_models:
                    engine.train_rf_model(n_estimators=int(rf_estimators), max_depth=int(rf_depth))
                if "XGBoost" in selected_models:
                    engine.train_xgboost_model()
                if "Logistic Regression" in selected_models:
                    engine.train_logistic_model()

                # ✅ IMPORTANT: Feature importance üret
                engine.analyze_feature_importance()

                metrics_rows = []

                def add_metrics(name: str, model):
                    if model is None:
                        st.warning(f"{name} selected but model is None (not installed or training failed).")
                        return
                    if engine.X_test is None or engine.X_test.empty:
                        st.warning("Test split is empty (not enough labeled data).")
                        return
                    y_pred = model.predict(engine.X_test)
                    metrics_rows.append(
                        {
                            "Model": name,
                            "Accuracy": accuracy_score(engine.y_test, y_pred),
                            "F1 Macro": f1_score(engine.y_test, y_pred, average="macro", zero_division=0),
                            "F1 Weighted": f1_score(engine.y_test, y_pred, average="weighted", zero_division=0),
                        }
                    )

                add_metrics("Random Forest", engine.rf_model)
                add_metrics("XGBoost", engine.xgb_model)
                add_metrics("Logistic Regression", engine.logreg_model)

                st.session_state.models_trained = True

            st.success("Training complete!")

            metrics_df = pd.DataFrame(metrics_rows)
            if not metrics_df.empty:
                st.subheader("Model performance")
                st.dataframe(metrics_df.sort_values("F1 Macro", ascending=False), use_container_width=True)

                st.subheader("Confusion matrices")
                tabs = st.tabs([r["Model"] for r in metrics_rows])

                label_vals = np.unique(engine.y_test)
                for tab, r in zip(tabs, metrics_rows):
                    with tab:
                        if r["Model"] == "Random Forest":
                            model = engine.rf_model
                        elif r["Model"] == "XGBoost":
                            model = engine.xgb_model
                        else:
                            model = engine.logreg_model

                        y_pred = model.predict(engine.X_test)
                        cm = confusion_matrix(engine.y_test, y_pred, labels=label_vals)

                        st.pyplot(plot_confusion_matrix(cm, label_vals))

                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"true_{x}" for x in label_vals],
                            columns=[f"pred_{x}" for x in label_vals],
                        )
                        st.dataframe(cm_df, use_container_width=True)

                st.subheader("Feature importance")
                fi_chart = feature_importance_chart(engine, top_n=20)
                if fi_chart is None:
                    st.info("Feature importance henüz hazır değil (RF/XGB eğitimi yapılmamış olabilir).")
                else:
                    st.altair_chart(fi_chart, use_container_width=True)


# -----------------------------
# PAGE 3: DECISIONS & EXPORT
# -----------------------------
elif page == "📊 Decisions & Export":
    st.title("📊 Decisions & Export ")

    engine: Optional[QualityDecisionSystem] = st.session_state.engine
    results_df = st.session_state.results_df

    if engine is None:
        st.warning("Upload a dataset first.")
    else:
        # ✅ Generate Decisions butonu burada
        st.subheader("Generate decisions")
        st.caption("After model training, it generates decisions and prepares the export table.")

        can_generate = st.session_state.models_trained is True
        if st.button("⚙️ Generate Decisions (process_all_data)", disabled=not can_generate):
            with st.spinner("Generating decisions..."):
                # process_all_data için df_model hazır olmalı
                if engine.df_model is None:
                    run_feature_engineering(engine)
                    engine.prepare_for_modeling()

                engine.process_all_data()
                st.session_state.results_df = engine.results_df

            st.success("Decisions generated!")

        results_df = st.session_state.results_df
        if results_df is None:
            st.warning("Decisions not yet generated. (Run Train & Evaluate in Modeling Studio first.)")
        else:
            st.subheader("Decision table preview")
            st.dataframe(safe_df_for_streamlit(results_df.head(50)), use_container_width=True)

            csv_export = results_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download full exported CSV",
                data=csv_export,
                file_name="qaisu_full_export.csv",
                mime="text/csv",
            )
