"""
Streamlit app — Mundial 2026 Predicciones
Run:  streamlit run app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mundial 2026 — Predicciones",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "Spain": "🇪🇸",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Germany": "🇩🇪", "Netherlands": "🇳🇱", "Portugal": "🇵🇹",
    "Belgium": "🇧🇪", "Colombia": "🇨🇴", "Uruguay": "🇺🇾", "Mexico": "🇲🇽",
    "United States": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵", "Morocco": "🇲🇦",
    "Senegal": "🇸🇳", "Croatia": "🇭🇷", "Switzerland": "🇨🇭",
    "Ecuador": "🇪🇨", "Australia": "🇦🇺", "Korea Republic": "🇰🇷",
    "Turkey": "🇹🇷", "Sweden": "🇸🇪", "Norway": "🇳🇴", "Austria": "🇦🇹",
    "Czechia": "🇨🇿", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Ghana": "🇬🇭", "Tunisia": "🇹🇳",
    "Egypt": "🇪🇬", "Algeria": "🇩🇿", "Saudi Arabia": "🇸🇦", "Iraq": "🇮🇶",
    "Iran": "🇮🇷", "South Africa": "🇿🇦", "Ivory Coast": "🇨🇮", "Congo DR": "🇨🇩",
    "Qatar": "🇶🇦", "Uzbekistan": "🇺🇿", "Jordan": "🇯🇴", "Panama": "🇵🇦",
    "Paraguay": "🇵🇾", "Haiti": "🇭🇹", "New Zealand": "🇳🇿",
    "Bosnia and Herzegovina": "🇧🇦", "Cabo Verde": "🇨🇻", "Curacao": "🇨🇼",
}

MODEL_LABELS = {
    "combined":    "Combinado ★",
    "form":        "Forma reciente",
    "elo":         "ELO histórico",
    "tournament":  "Historial WC",
    "fifa_rank":   "Ranking FIFA",
    "host_boost":  "Host boost",
    "poisson":     "Poisson (goles)",
    "market_odds": "Apuestas 2026",
}

STAGE_LABELS = {
    "p_r32":      "Round of 32",
    "p_r16":      "Octavos",
    "p_qf":       "Cuartos",
    "p_sf":       "Semis",
    "p_final":    "Final",
    "p_champion": "Campeón 🏆",
}

HISTORICAL_DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳")


def flabel(team: str) -> str:
    return f"{flag(team)} {team}"


# ---------------------------------------------------------------------------
# Data loading — CSVs (pre-computed, no heavy computation needed)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csvs() -> dict:
    files = {
        "champion":     "output/champion_probabilities.csv",
        "simulation":   "output/simulation_results.csv",
        "matches":      "output/match_predictions.csv",
        "elo":          "output/elo_ratings.csv",
        "backtest":     "output/backtest_metrics.csv",
        "backtest_year":"output/backtest_by_year.csv",
        "teams":        "data/teams.csv",
    }
    return {k: pd.read_csv(v) for k, v in files.items() if os.path.exists(v)}


# ---------------------------------------------------------------------------
# Model loading — expensive, cached for the whole session
# ---------------------------------------------------------------------------

def ensure_historical_data() -> bool:
    path = os.path.join("data", "historical_results.csv")
    if os.path.exists(path):
        return True
    os.makedirs("data", exist_ok=True)
    with st.spinner("Descargando datos históricos (~8 MB, solo la primera vez)…"):
        try:
            r = requests.get(HISTORICAL_DATA_URL, timeout=90)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception as e:
            st.error(f"No se pudo descargar el historial: {e}")
            return False


@st.cache_resource(show_spinner="Cargando modelos (puede tardar ~40 s la primera vez)…")
def load_models():
    from profiles import build_all_profiles, CANONICAL_NAME_MAP
    from elo import calculate_elo

    teams_df = pd.read_csv("data/teams.csv")
    results_df = pd.read_csv("data/historical_results.csv", parse_dates=["date"])
    results_df["home_team"] = results_df["home_team"].replace(CANONICAL_NAME_MAP)
    results_df["away_team"] = results_df["away_team"].replace(CANONICAL_NAME_MAP)

    elo_raw = calculate_elo(results_df)
    profiles = build_all_profiles(teams_df, results_df, elo_raw)
    return profiles, sorted(teams_df["team"].tolist())


# ---------------------------------------------------------------------------
# Tab: Campeón
# ---------------------------------------------------------------------------

def tab_campeon(data: dict):
    st.header("🏆 Probabilidades de Campeón")

    champ = data.get("champion")
    if champ is None:
        st.warning("No hay datos. Ejecutá `python src/predict.py` primero.")
        return

    model_cols = [c for c in champ.columns if c not in ("team", "avg_champion_prob")]

    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox(
            "Modelo",
            ["avg_champion_prob"] + model_cols,
            format_func=lambda x: "Promedio (todos los modelos)" if x == "avg_champion_prob"
                                   else MODEL_LABELS.get(x, x),
        )
    with c2:
        top_n = st.slider("Top N", 5, 48, 20)

    plot_df = champ.nlargest(top_n, sel).copy()
    plot_df["label"] = plot_df["team"].apply(flabel)
    plot_df["pct"] = (plot_df[sel] * 100).round(1)

    fig = px.bar(
        plot_df.sort_values(sel),
        x=sel, y="label", orientation="h",
        text="pct",
        color=sel, color_continuous_scale="Blues",
        labels={sel: "Probabilidad", "label": ""},
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        height=max(380, top_n * 30),
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
        margin=dict(l=10, r=70, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabla completa — todos los modelos"):
        disp = champ.copy()
        disp.insert(0, " ", disp["team"].apply(flag))
        for col in model_cols + ["avg_champion_prob"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:.1%}")
        disp = disp.rename(columns={
            **{c: MODEL_LABELS.get(c, c) for c in model_cols},
            "avg_champion_prob": "Promedio",
            "team": "Equipo",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Etapas (heatmap)
# ---------------------------------------------------------------------------

def tab_etapas(data: dict):
    st.header("📊 Probabilidades por Etapa")

    sim = data.get("simulation")
    if sim is None:
        st.warning("No hay datos de simulación.")
        return

    models = sim["model"].unique().tolist()
    sel = st.selectbox(
        "Modelo",
        models,
        format_func=lambda x: MODEL_LABELS.get(x, x),
        index=models.index("combined") if "combined" in models else 0,
        key="etapas_model",
    )

    df = sim[sim["model"] == sel].sort_values("p_champion", ascending=False).copy()
    stage_cols = list(STAGE_LABELS.keys())
    df["label"] = df["team"].apply(flabel)

    heat = df.set_index("label")[stage_cols].rename(columns=STAGE_LABELS)

    fig = px.imshow(
        heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=".0%",
        labels=dict(color="Prob."),
    )
    fig.update_layout(
        height=1300,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
    )
    fig.update_traces(textfont_size=9)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Grupos
# ---------------------------------------------------------------------------

def tab_grupos(data: dict):
    st.header("⚽ Grupos — Predicciones")

    matches = data.get("matches")
    if matches is None:
        st.warning("No hay datos de partidos.")
        return

    from simulate import GROUPS

    models = matches["model"].unique().tolist()
    sel = st.selectbox(
        "Modelo",
        models,
        format_func=lambda x: MODEL_LABELS.get(x, x),
        index=models.index("combined") if "combined" in models else 0,
        key="grupos_model",
    )

    group_letters = sorted(GROUPS.keys())
    for row_start in range(0, len(group_letters), 3):
        row_groups = group_letters[row_start:row_start + 3]
        cols = st.columns(3)
        for col, grp in zip(cols, row_groups):
            with col:
                st.markdown(f"### Grupo {grp}")
                for t in GROUPS[grp]:
                    st.write(f"{flag(t)} {t}")

                gm = matches[
                    (matches["group"] == grp) & (matches["model"] == sel)
                ].sort_values("matchday")

                rows_d = []
                for _, r in gm.iterrows():
                    rows_d.append({
                        "MD": int(r["matchday"]),
                        "Local": flabel(r["home_team"]),
                        "Local %": f"{r['p_home_win']:.0%}",
                        "Empate": f"{r['p_draw']:.0%}",
                        "Visita %": f"{r['p_away_win']:.0%}",
                        "Visita": flabel(r["away_team"]),
                    })
                st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Predictor (requiere modelos cargados)
# ---------------------------------------------------------------------------

def tab_predictor():
    st.header("🎯 Predictor de Partidos")
    st.caption("Elegí dos equipos y ve las probabilidades según cada modelo.")

    if not ensure_historical_data():
        return

    with st.spinner("Cargando modelos…"):
        profiles, all_teams = load_models()

    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        team_a = st.selectbox(
            "Equipo A", all_teams,
            index=all_teams.index("Argentina") if "Argentina" in all_teams else 0,
            key="pred_a",
        )
    with c2:
        st.markdown("<br><br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with c3:
        team_b = st.selectbox(
            "Equipo B", all_teams,
            index=all_teams.index("Brazil") if "Brazil" in all_teams else 1,
            key="pred_b",
        )

    if team_a == team_b:
        st.warning("Elegí dos equipos distintos.")
        return

    rows = []
    for pname, profile in profiles.items():
        pa, pd_, pb = profile.match_proba(team_a, team_b, neutral=True)
        rows.append({
            "model_key": pname,
            "Modelo": MODEL_LABELS.get(pname, pname),
            "team_a": pa,
            "draw": pd_,
            "team_b": pb,
        })
    rdf = pd.DataFrame(rows)

    # Bar chart
    fig = go.Figure()
    for col, color, label in [
        ("team_a", "#1f77b4", flabel(team_a)),
        ("draw",   "#aec7e8", "Empate"),
        ("team_b", "#ff7f0e", flabel(team_b)),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=rdf["Modelo"],
            y=rdf[col],
            marker_color=color,
            text=[f"{v:.0%}" for v in rdf[col]],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        height=420,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=50, b=90, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    disp = rdf[["Modelo"]].copy()
    disp[flabel(team_a)] = rdf["team_a"].apply(lambda x: f"{x:.1%}")
    disp["Empate"]        = rdf["draw"].apply(lambda x: f"{x:.1%}")
    disp[flabel(team_b)]  = rdf["team_b"].apply(lambda x: f"{x:.1%}")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Backtest
# ---------------------------------------------------------------------------

def tab_backtest(data: dict):
    st.header("📈 Backtest — Precisión histórica")
    st.caption("256 partidos de los Mundiales 2010 · 2014 · 2018 · 2022 (sin data leakage).")

    metrics = data.get("backtest")
    if metrics is None:
        st.info("Ejecutá `python src/backtest.py` para generar métricas.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Brier Score (↓ mejor)")
        df_b = metrics.copy().sort_values("avg_brier")
        df_b["label"] = df_b["model"].apply(lambda x: MODEL_LABELS.get(x, x))
        fig = px.bar(
            df_b, x="avg_brier", y="label", orientation="h",
            color="avg_brier", color_continuous_scale="RdYlGn_r",
            text=df_b["avg_brier"].apply(lambda x: f"{x:.4f}"),
            labels={"avg_brier": "", "label": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=320,
                          margin=dict(t=10, b=10, l=10, r=70))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Accuracy (↑ mejor)")
        df_a = metrics.copy().sort_values("accuracy", ascending=False)
        df_a["label"] = df_a["model"].apply(lambda x: MODEL_LABELS.get(x, x))
        fig2 = px.bar(
            df_a, x="accuracy", y="label", orientation="h",
            color="accuracy", color_continuous_scale="Blues",
            text=df_a["accuracy"].apply(lambda x: f"{x:.1%}"),
            labels={"accuracy": "", "label": ""},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=320,
                           xaxis_tickformat=".0%",
                           margin=dict(t=10, b=10, l=10, r=70))
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    tbl = metrics.copy()
    tbl["model"] = tbl["model"].apply(lambda x: MODEL_LABELS.get(x, x))
    tbl["accuracy"] = tbl["accuracy"].apply(lambda x: f"{x:.1%}")
    tbl["avg_brier"] = tbl["avg_brier"].apply(lambda x: f"{x:.5f}")
    tbl["avg_logloss"] = tbl["avg_logloss"].apply(lambda x: f"{x:.5f}")
    tbl = tbl.rename(columns={
        "model": "Modelo", "avg_brier": "Brier ↓",
        "avg_logloss": "Log Loss ↓", "accuracy": "Accuracy ↑",
        "n_matches": "Partidos",
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # By year
    byyear = data.get("backtest_year")
    if byyear is not None:
        st.subheader("Brier Score por año")
        by = byyear.copy()
        by["model"] = by["model"].apply(lambda x: MODEL_LABELS.get(x, x))
        for col in [2010, 2014, 2018, 2022, "avg"]:
            if col in by.columns:
                by[col] = by[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        st.dataframe(by, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Datos ELO
# ---------------------------------------------------------------------------

def tab_datos(data: dict):
    st.header("🔍 Datos y Rankings")

    elo = data.get("elo")
    teams = data.get("teams")

    if elo is not None:
        st.subheader("Ratings ELO — equipos del Mundial 2026")

        confs = sorted(elo["confederation"].dropna().unique()) if "confederation" in elo.columns else []
        sel_conf = st.multiselect("Filtrar por confederación", confs)

        df = elo.copy()
        if sel_conf:
            df = df[df["confederation"].isin(sel_conf)]

        df.insert(0, " ", df["team"].apply(flag))
        cols_show = [" ", "team", "elo", "elo_rank", "group", "confederation", "fifa_rank", "fifa_pts"]
        cols_show = [c for c in cols_show if c in df.columns]
        df = df[cols_show].rename(columns={
            "team": "Equipo", "elo": "ELO", "elo_rank": "Pos. ELO",
            "group": "Grupo", "confederation": "Conf.",
            "fifa_rank": "Rank FIFA", "fifa_pts": "Pts FIFA",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ELO scatter by confederation
        if "confederation" in elo.columns:
            fig = px.scatter(
                elo, x="fifa_pts", y="elo",
                color="confederation",
                hover_name="team",
                text="team",
                labels={"fifa_pts": "Puntos FIFA", "elo": "Rating ELO"},
                height=500,
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("⚽ Mundial 2026 — Predicciones")
    st.caption(
        "7 modelos: ELO · Forma reciente · Historial WC · Ranking FIFA · "
        "Poisson (goles) · Apuestas · **Combinado** (blend optimizado por backtest)"
    )

    data = load_csvs()

    tabs = st.tabs([
        "🏆 Campeón",
        "📊 Etapas",
        "⚽ Grupos",
        "🎯 Predictor",
        "📈 Backtest",
        "🔍 Datos",
    ])

    with tabs[0]:
        tab_campeon(data)
    with tabs[1]:
        tab_etapas(data)
    with tabs[2]:
        tab_grupos(data)
    with tabs[3]:
        tab_predictor()
    with tabs[4]:
        tab_backtest(data)
    with tabs[5]:
        tab_datos(data)


if __name__ == "__main__":
    main()
