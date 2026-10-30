# -*- coding: utf-8 -*-
"""
Deferral Explainer: explaining why a conformal selective prediction model abstains.

Data product of the MDS research project "An Uncertainty Attribution Model for Explaining Model
Abstention in Clinical Risk Prediction" (YANG PEIPEI, Universiti Malaya).

Everything on the page is computed live from the fitted models:
  * the recalibrated LightGBM probability p of the selected (or edited) case,
  * the class-conditional conformal prediction set at the chosen target coverage, with thresholds
    computed from the calibration set,
  * the ambiguity score A = -|logit p - logit pi| and the abstention propensity R over 200
    calibration bootstraps,
  * grouped KernelSHAP attributions of A (proposed) and of p (conventional), with the coalition
    sample and background used in the study,
  * a what-if: how often the deferral is resolved when the top-k named feature groups are replaced
    by background values.
"""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent / "assets"
DATASETS = {"Diabetes": "diabetes", "Sepsis (hospital B)": "sepsis", "OULAD": "oulad"}
COVERAGE = {"95%": 0.05, "90%": 0.10, "80%": 0.20}
EXAMPLES = {"diabetes": [26747, 89299]}
RED, GREY, AMBER = "#c0392b", "#7f8c8d", "#e9a23b"

st.set_page_config(page_title="Deferral Explainer", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- model and data
def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


@st.cache_resource(show_spinner="Loading model and data ...")
def load(name):
    d = ROOT / name
    meta = json.load(open(d / "meta.json"))
    meta["nice"] = [n[:1].upper() + n[1:] for n in meta["nice"]]
    z = np.load(d / "data.npz")
    data = {k: z[k] for k in z.files}
    booster = lgb.Booster(model_file=str(d / "model.txt"))
    return meta, data, booster


class Engine:
    """All live computations for one dataset."""

    def __init__(self, name):
        self.meta, self.d, self.b = load(name)
        self.a, self.c = self.meta["platt"]
        self.lpi = self.meta["logit_pi"]
        self.G = self.d["G"].astype(int)
        self.lo_b = np.sort(self.d["lohi"][:, 0])
        self.hi_b = np.sort(self.d["lohi"][:, 1])

    def p(self, X):
        return expit(self.a * self.b.predict(np.asarray(X, dtype=np.float32), raw_score=True) + self.c)

    def band(self, alpha):
        """Class-conditional split conformal thresholds (LAC score, Sadinle et al., 2019): the set is
        {0, 1} exactly when lo <= p <= hi."""
        p, y = self.d["p_cal"], self.d["y_cal"]
        q = []
        for s in (np.sort(p[y == 0]), np.sort(1 - p[y == 1])):
            k = int(np.ceil((len(s) + 1) * (1 - alpha)))
            q.append(s[min(k, len(s)) - 1])
        return 1 - q[1], q[0]

    def pred_set(self, p, alpha):
        lo, hi = self.band(alpha)
        s = ([0] if p <= hi else []) + ([1] if p >= lo else [])
        return s

    def targets(self, X):
        p = self.p(X)
        A = -np.abs(logit(p) - self.lpi)
        R = (np.searchsorted(self.lo_b, p, side="right") - np.searchsorted(self.hi_b, p, side="left")) / len(self.lo_b)
        return np.column_stack([p, A, R])

    def explain(self, x):
        """Grouped KernelSHAP with the study's coalition sample and background (efficiency exact)."""
        Z, Bg, G = self.d["Z"], self.d["X_bg"], self.G
        mask = Z[:, G]
        rows = np.where(mask[:, None, :], x[None, None, :], Bg[None, :, :]).reshape(-1, x.shape[0])
        vals = self.targets(rows).reshape(len(Z), len(Bg), -1).mean(1)
        v0, v1 = self.targets(Bg).mean(0), self.targets(x[None, :])[0]
        Zf = Z.astype(float)
        delta = v1 - v0
        yv = (vals - v0) - Zf[:, [-1]] * delta
        A = Zf[:, :-1] - Zf[:, [-1]]
        head, *_ = np.linalg.lstsq(A, yv, rcond=None)
        phi = np.vstack([head, delta - head.sum(0)])
        return phi, v1                                   # phi: [groups, (p, A, R)]

    def resolution(self, x, groups, alpha):
        """Share of background replacements of the given groups after which {0,1} becomes a singleton."""
        lo, hi = self.band(alpha)
        Bg = self.d["X_bg"]
        rows = np.repeat(x[None, :], len(Bg), 0)
        m = np.isin(self.G, groups)
        rows[:, m] = Bg[:, m]
        pp = self.p(rows)
        return float(((pp < lo) | (pp > hi)).mean())


@st.cache_data(show_spinner="Computing Shapley attributions (KernelSHAP, 1,024 coalitions x 100 background cases) ...",
               max_entries=200)
def explain_cached(name, xbytes, ncols):
    x = np.frombuffer(xbytes, dtype=np.float32).reshape(ncols)
    return ENG[name].explain(x)


ENG = {n: Engine(n) for n in DATASETS.values()}


# --------------------------------------------------------------------------- value display / editing
def describe(meta, x):
    """Readable value of every feature group of a (possibly edited) feature vector."""
    cols, groups = meta["cols"], meta["groups"]
    G = ENG_CUR.G
    out = []
    for j, g in enumerate(groups):
        idx = np.flatnonzero(G == j)
        names = [cols[i] for i in idx]
        if all("=" in c for c in names):
            out.append(names[int(np.argmax(x[idx]))].split("=", 1)[1])
            continue
        last = [k for k, c in zip(idx, names) if c == f"{g}_last"]
        cnt = [k for k, c in zip(idx, names) if c == f"{g}_n"]
        if cnt:                                          # sepsis vital sign or laboratory value
            n = float(x[cnt[0]])
            if n == 0 or np.isnan(n):
                out.append("not measured in 24 h")
            elif last and not np.isnan(float(x[last[0]])):
                v = float(x[last[0]])
                out.append(f"last {v:.0f} ({n:.0f} obs)" if abs(v - round(v)) < 1e-6 else f"last {v:.2f} ({n:.0f} obs)")
            else:
                out.append(f"{n:.0f} measurements")
            continue
        parts = []
        for i, c in zip(idx, names):
            v = float(x[i])
            if c in meta["log1p_cols"]:
                v = np.expm1(v)
            if np.isnan(v):
                parts.append("missing" if len(idx) == 1 else f"{c[len(g) + 1:] or c}: missing")
            else:
                s = f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
                parts.append(s if len(idx) == 1 else f"{c[len(g) + 1:] or c}: {s}")
        out.append(", ".join(parts))
    return out


def edit_form(meta, x, key):
    """Widgets for every feature group; returns the edited vector when submitted."""
    cols, groups, nice = meta["cols"], meta["groups"], meta["nice"]
    G = ENG_CUR.G
    new = x.copy()
    with st.form(key):
        st.caption("Change any recorded value and press **Recompute**. The model, the conformal set and both "
                   "attributions are recomputed for the edited case. Leave a number empty to mark it as missing.")
        ncol = 3
        cells = st.columns(ncol)
        for j, g in enumerate(groups):
            idx = np.flatnonzero(G == j)
            names = [cols[i] for i in idx]
            box = cells[j % ncol]
            if all("=" in c for c in names):
                levels = [c.split("=", 1)[1] for c in names]
                cur = int(np.argmax(x[idx]))
                choice = box.selectbox(nice[j], levels, index=cur, key=f"{key}_{j}")
                new[idx] = 0
                new[idx[levels.index(choice)]] = 1
                continue
            for i, c in zip(idx, names):
                v = float(x[i])
                is_log = c in meta["log1p_cols"]
                shown = None if np.isnan(v) else float(np.expm1(v) if is_log else v)
                label = nice[j] if len(idx) == 1 else f"{nice[j]} · {c[len(g) + 1:] or c}"
                val = box.number_input(label, value=shown, key=f"{key}_{j}_{i}", format="%.3f")
                new[i] = np.nan if val is None else (np.log1p(max(val, 0.0)) if is_log else val)
        done = st.form_submit_button("Recompute", type="primary")
    return new if done else None


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Deferral Explainer")
st.sidebar.caption("Why does the model hand this case back to a human?")
ds_label = st.sidebar.radio("Dataset", list(DATASETS), horizontal=False)
name = DATASETS[ds_label]
ENG_CUR = ENG[name]
meta, D = ENG_CUR.meta, ENG_CUR.d
cov_label = st.sidebar.radio("Target coverage (1 − α)", list(COVERAGE), index=1, horizontal=True)
alpha = COVERAGE[cov_label]
lo, hi = ENG_CUR.band(alpha)

p_cases = ENG_CUR.p(D["X_cases"])
z_cases = logit(p_cases) - meta["logit_pi"]
deferred = (p_cases >= lo) & (p_cases <= hi)
show = st.sidebar.radio("Cases", ["Deferred", "Committed", "All"], horizontal=True)
mask = deferred if show == "Deferred" else (~deferred if show == "Committed" else np.ones_like(deferred))
order = np.argsort(z_cases)
options = [int(i) for i in order if mask[i]]
ids = D["case_id"]


def label_of(i):
    s = ENG_CUR.pred_set(p_cases[i], alpha)
    return f"#{ids[i]}   p = {p_cases[i]:.3f}   set {{{', '.join(map(str, s))}}}"


ex = [e for e in EXAMPLES.get(name, []) if e in set(ids.tolist())]
if ex:
    st.sidebar.caption("Worked examples from the report")
    bcols = st.sidebar.columns(len(ex))
    for c, e in zip(bcols, ex):
        if c.button(f"#{e}", key=f"ex_{name}_{e}"):
            st.session_state[f"case_{name}"] = int(np.flatnonzero(ids == e)[0])
            st.session_state.pop(f"edited_{name}", None)
if not options:
    st.sidebar.warning("No cases in this group at this coverage.")
    st.stop()
default = st.session_state.get(f"case_{name}", options[len(options) // 2])
if default not in options:
    default = options[0]
pick = st.sidebar.selectbox(f"{len(options)} cases, ordered by position in the band", options,
                            index=options.index(default), format_func=label_of)
if pick != st.session_state.get(f"case_{name}"):
    st.session_state[f"case_{name}"] = pick
    st.session_state.pop(f"edited_{name}", None)
st.sidebar.divider()
st.sidebar.caption(f"Model: LightGBM with Platt recalibration · {meta['label']} · test AUC {meta['auc']:.3f}. "
                   "Research prototype, not for clinical use.")

# --------------------------------------------------------------------------- current case
i = st.session_state[f"case_{name}"]
x0 = D["X_cases"][i].astype(np.float32)
x = st.session_state.get(f"edited_{name}", x0)
edited = f"edited_{name}" in st.session_state
phi, v1 = explain_cached(name, x.tobytes(), x.shape[0])
p, A, R = float(v1[0]), float(v1[1]), float(v1[2])
s = ENG_CUR.pred_set(p, alpha)
is_def = len(s) == 2
zpos = float(logit(np.array([p]))[0] - meta["logit_pi"])
vals = describe(meta, x)

tab1, tab2, tab3 = st.tabs(["Explain a case", "Evidence from the study", "About"])

with tab1:
    head = f"{meta['label']} · {meta['unit']} #{ids[i]}" + ("  ·  edited" if edited else "")
    st.caption(head.upper())
    verdict = "Deferred to a human" if is_def else ("Committed: " + (meta["pos"] if s == [1] else meta["neg"])
                                                     if len(s) == 1 else "Empty set")
    st.markdown(f"### {verdict}")
    st.markdown(f"Prediction set at {cov_label} target coverage: **{{{', '.join(map(str, s))}}}**")
    cards = [("Predicted probability p", f"{p:.3f}"), ("Ambiguity score A", f"{A:.3f}"),
             (f"Deferred in calibration resamples (α = {meta['alpha_R']:.2f})", f"{R * 100:.0f}% of {meta['n_boot']}"),
             ("Observed outcome", meta["pos"] if int(D["y_cases"][i]) == 1 else meta["neg"])]
    for col, (k, v) in zip(st.columns(4), cards):
        col.markdown(f"<div style='font-size:0.82rem;color:#5f6b76'>{k}</div>"
                     f"<div style='font-size:1.35rem;font-weight:600'>{v}</div>", unsafe_allow_html=True)
    if edited:
        if st.button("Reset to the recorded values"):
            st.session_state.pop(f"edited_{name}", None)
            st.rerun()

    # ---- where the case sits
    st.subheader("Where the case sits")
    zl, zh = float(logit(np.array([lo]))[0] - meta["logit_pi"]), float(logit(np.array([hi]))[0] - meta["logit_pi"])
    span = max(3.0, abs(zpos) + 0.5)
    base = alt.Chart(pd.DataFrame({"a": [-span], "b": [span]})).mark_rect(color="#eef1f4", height=26).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[-span, span]),
                title=f"log-odds of '{meta['pos']}' relative to the training prevalence (logit p − logit π)"),
        x2="b:Q")
    bandc = alt.Chart(pd.DataFrame({"a": [zl], "b": [zh], "t": ["defer zone"]})).mark_rect(
        color=AMBER, opacity=0.45, height=26).encode(x="a:Q", x2="b:Q", tooltip=["t"])
    prev = alt.Chart(pd.DataFrame({"z": [0], "t": ["prevalence"]})).mark_tick(
        color="#333", thickness=2, size=36).encode(x="z:Q", tooltip=["t"])
    prevt = alt.Chart(pd.DataFrame({"z": [0], "t": ["prevalence"]})).mark_text(
        dy=-24, fontSize=11, color="#333").encode(x="z:Q", text="t:N")
    me = alt.Chart(pd.DataFrame({"z": [zpos], "t": [f"this case: p = {p:.3f}"]})).mark_point(
        shape="triangle-down", size=260, filled=True, color=RED, yOffset=-16).encode(x="z:Q", tooltip=["t"])
    st.altair_chart((base + bandc + prev + prevt + me).properties(height=90), width="stretch")
    st.caption(f"The amber zone is where both labels stay in the prediction set at {cov_label} target coverage "
               f"(p between {lo:.3f} and {hi:.3f}); the dark line marks the training prevalence "
               f"({100 / (1 + np.exp(-meta['logit_pi'])):.1f}%). Changing the coverage moves the zone; the ambiguity "
               "attribution below does not change, because A does not use the calibration set.")

    # ---- why the model cannot decide
    st.subheader("Why the model cannot decide")
    n_top = 8
    top = np.argsort(-np.abs(phi[:, 1]))[:n_top]
    rows = []
    for j in top:
        rows.append({"Feature group": meta["nice"][j], "Value": vals[j],
                     "Attribution": "Ambiguity A (proposed)", "SHAP": float(phi[j, 1])})
        rows.append({"Feature group": meta["nice"][j], "Value": vals[j],
                     "Attribution": "Probability p (conventional)", "SHAP": float(phi[j, 0])})
    df = pd.DataFrame(rows)
    # normalise each attribution to its largest absolute value among the shown groups, for comparability
    for t in df["Attribution"].unique():
        m = df["Attribution"] == t
        df.loc[m, "Normalised"] = df.loc[m, "SHAP"] / max(df.loc[m, "SHAP"].abs().max(), 1e-12)
    order_names = [meta["nice"][j] for j in top]
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Feature group:N", sort=order_names, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("Attribution:N"),
        x=alt.X("Normalised:Q", title="Normalised Shapley value (+ raises the quantity)",
                scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("Attribution:N", scale=alt.Scale(range=[RED, GREY]),
                        legend=alt.Legend(orient="top", labelLimit=400, title=None)),
        tooltip=["Feature group", "Value", "Attribution", alt.Tooltip("SHAP:Q", format=".4f")])
    left = right = st.container()
    left.altair_chart(chart.properties(height=40 * n_top), width="stretch")
    tbl = pd.DataFrame({"Feature group": order_names, "Value": [vals[j] for j in top],
                        "SHAP of A": [f"{phi[j, 1]:+.4f}" for j in top],
                        "SHAP of p": [f"{phi[j, 0]:+.4f}" for j in top]})
    right.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Feature group": st.column_config.TextColumn(width="medium"),
        "Value": st.column_config.TextColumn(width="medium"),
        "SHAP of A": st.column_config.TextColumn(width="small"),
        "SHAP of p": st.column_config.TextColumn(width="small")})
    # where the two explanations disagree
    msgs = []
    for j in top[:5]:
        a_, p_ = phi[j, 1], phi[j, 0]
        if abs(a_) < 1e-4:
            continue
        if np.sign(a_) != np.sign(p_) and abs(p_) > 1e-4:
            msgs.append(f"**{meta['nice'][j]}** ({vals[j]}) {'raises' if p_ > 0 else 'lowers'} the predicted risk but "
                        f"{'adds to' if a_ > 0 else 'reduces'} the ambiguity.")
        elif abs(a_) > 5 * abs(p_):
            msgs.append(f"**{meta['nice'][j]}** ({vals[j]}) {'adds to' if a_ > 0 else 'reduces'} the ambiguity far "
                        "more than it changes the predicted risk.")
    if msgs:
        right.markdown("**Where the two explanations disagree**\n\n" + "\n\n".join("- " + m for m in msgs[:3]))
    st.caption("Red: grouped KernelSHAP of the ambiguity score A (positive values pull the case into the defer zone). "
               "Grey: conventional SHAP of the predicted probability p (positive values raise the risk). Both use "
               "1,024 coalitions and 100 background cases, as in the study.")

    # ---- what-if
    st.subheader("What would resolve the deferral?")
    if is_def:
        rankA = [int(j) for j in np.argsort(-phi[:, 1])]
        rankP = [int(j) for j in np.argsort(-np.abs(phi[:, 0]))]
        ks = list(range(1, 6))
        wf = pd.DataFrame({
            "k": ks * 2,
            "Resolution rate": [ENG_CUR.resolution(x, rankA[:k], alpha) for k in ks] +
                               [ENG_CUR.resolution(x, rankP[:k], alpha) for k in ks],
            "Ranking": ["Ambiguity attribution (proposed)"] * 5 + ["Conventional SHAP"] * 5})
        line = alt.Chart(wf).mark_line(point=True).encode(
            x=alt.X("k:O", title="Top-k feature groups replaced by typical (background) values"),
            y=alt.Y("Resolution rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                    title="Deferral resolved"),
            color=alt.Color("Ranking:N", scale=alt.Scale(range=[RED, GREY]),
                            legend=alt.Legend(orient="top", labelLimit=400, title=None)),
            tooltip=["Ranking", "k", alt.Tooltip("Resolution rate:Q", format=".0%")])
        st.altair_chart(line.properties(height=260), width="stretch")
        st.caption("For each of the 100 background cases, the top-k groups of this case are replaced by that case's "
                   "values and the conformal set is recomputed; the curve shows how often the set becomes a single "
                   "label. This is the set-size faithfulness criterion of the study, applied to one case.")
    else:
        st.info("The case is not deferred at this coverage, so there is no deferral to resolve. Choose a deferred case "
                "or a higher target coverage.")

    with st.expander("Edit this case and recompute (what-if)", expanded=False):
        newx = edit_form(meta, x, f"form_{name}_{i}")
        if newx is not None:
            st.session_state[f"edited_{name}"] = newx.astype(np.float32)
            st.rerun()

with tab2:
    st.subheader(f"{meta['label']}: what the study found")
    c1, c2, c3 = st.columns(3)
    c1.metric("LightGBM test AUC", f"{meta['auc']:.3f}")
    c2.metric("Minority coverage at 90%: marginal → class-conditional",
              f"{meta['cov_pos_marginal']:.2f} → {meta['cov_pos_class']:.2f}")
    c3.metric(f"Cases deferred at {cov_label} (these 600 cases)", f"{deferred.mean() * 100:.0f}%")
    f = meta["faith"]
    fd = pd.DataFrame([{"k": k, "Ranking": r, "Resolution rate": v}
                       for r, vs in f.items() for k, v in enumerate(vs)])
    st.markdown("**Faithfulness (set-size criterion) over the 400 deferred cases explained in the study**")
    st.altair_chart(alt.Chart(fd).mark_line(point=True).encode(
        x=alt.X("k:O", title="Top-k feature groups masked"),
        y=alt.Y("Resolution rate:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Ranking:N", legend=alt.Legend(orient="top", labelLimit=400, title=None))).properties(height=280),
        width="stretch")
    g = meta["global"]
    gd = pd.DataFrame({"group": g["group"], "A": g["A"], "p": g["p"]})
    gd["Feature group"] = [meta["nice"][meta["groups"].index(x)] if x in meta["groups"] else x for x in gd.group]
    gd = gd.sort_values("A", ascending=False).head(12)
    gl = gd.melt(id_vars="Feature group", value_vars=["A", "p"], var_name="t", value_name="Share")
    gl["Attribution"] = gl.t.map({"A": "Ambiguity A (drives deferral)", "p": "Probability p (drives risk)"})
    st.markdown("**Global attribution over the deferred cases: what drives deferral versus what drives risk**")
    st.altair_chart(alt.Chart(gl).mark_bar().encode(
        y=alt.Y("Feature group:N", sort=list(gd["Feature group"]), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset="Attribution:N", x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share of mean |SHAP|"),
        color=alt.Color("Attribution:N", scale=alt.Scale(range=[RED, GREY]),
                        legend=alt.Legend(orient="top", labelLimit=400, title=None))
    ).properties(height=34 * len(gd)), width="stretch")

with tab3:
    st.markdown("""
**What this application does.** When a conformal selective prediction model abstains, its prediction set contains both
labels and the case is handed to a human. This application explains *why* a case is deferred, next to what conventional
SHAP says about the predicted risk.

**Computed live for every case or edited case**
- the recalibrated LightGBM probability *p*;
- the class-conditional split conformal prediction set at the chosen target coverage (LAC score; thresholds from the
  calibration set);
- the ambiguity score *A* = −|logit *p* − logit π| and the abstention propensity *R* (share of 200 calibration
  bootstraps under which the case is deferred, α = 0.10);
- grouped KernelSHAP attributions of *A* and of *p* (1,024 coalitions, 100 background cases, exact efficiency);
- the resolution rate after replacing the top-k feature groups by background values.

**Data.** Diabetes 130-US Hospitals (UCI, CC BY 4.0); PhysioNet/CinC Challenge 2019 sepsis data (hospital B
patients, model calibrated on hospital A); Open University Learning Analytics Dataset (CC BY 4.0). Each dataset offers
the 600 held-out cases explained in the study (400 deferred and 200 committed at 90% coverage).

**Source.** YANG PEIPEI (2026). *An Uncertainty Attribution Model for Explaining Model Abstention in Clinical Risk
Prediction.* Master of Data Science research report, Universiti Malaya. Supervisor: Assoc. Prof. Dr. Maizatul Akmar
binti Ismail.

*Research prototype. It is not a medical device and must not be used for clinical decisions.*
""")
