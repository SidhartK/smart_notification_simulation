import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    import math
    import random
    import json

    return go, json, math, mo, pl, random


@app.cell
def _(mo):
    get_seed, set_seed = mo.state(42)

    return get_seed, set_seed


@app.cell
def _(mo, random, set_seed):
    freq_slider = mo.ui.slider(
        1, 60, step=1, value=12, label="Poll freq (per hour)", show_value=True
    )
    diff_slider = mo.ui.slider(
        0.3, 3.0, step=0.1, value=1.2, label="Difficulty", show_value=True
    )
    steep_slider = mo.ui.slider(
        0.02, 0.3, step=0.01, value=0.08, label="Steepness (k)", show_value=True
    )
    infl_slider = mo.ui.slider(
        0.2, 0.9, step=0.05, value=0.55, label="Inflection point", show_value=True
    )
    tdur_slider = mo.ui.slider(
        20, 90, step=5, value=45, label="Task duration (min)", show_value=True
    )
    gap_slider = mo.ui.slider(
        0.05, 0.5, step=0.05, value=0.25, label="Accept gap (below thresh)", show_value=True
    )
    reseed_btn = mo.ui.button(
        label="Re-randomise seed",
        on_click=lambda _: set_seed(random.randint(0, 10**9)),
    )
    mo.vstack([
        mo.md("### Engine Parameters"),
        freq_slider,
        diff_slider,
        steep_slider,
        infl_slider,
        tdur_slider,
        gap_slider,
        reseed_btn,
    ])
    return (
        diff_slider,
        freq_slider,
        gap_slider,
        infl_slider,
        steep_slider,
        tdur_slider,
    )


@app.cell
def _(
    diff_slider,
    freq_slider,
    gap_slider,
    get_seed,
    infl_slider,
    math,
    steep_slider,
    tdur_slider,
):
    def _run_sim(freq, diff, k, infl, tdur, gap, seed):
        """Port of the JS simulation: LCG RNG, sigmoid desire, fixed-freq polling."""

        def make_rng(s):
            state = [s]
            def rng():
                state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
                return state[0] / 0xFFFFFFFF
            return rng

        def desire(t, T, k_, diff_, infl_):
            mid = infl_ * T
            def sig(x):
                return 1 / (1 + math.exp(-x))
            raw = sig(k_ * diff_ * (t - mid))
            lo = sig(-k_ * diff_ * mid)
            hi = sig(k_ * diff_ * (T - mid))
            return (raw - lo) / (hi - lo)

        THRESHOLD = 0.85
        ACCEPT_FLOOR = THRESHOLD - gap
        DAY_MIN = 480
        RES = 2
        STEPS = DAY_MIN // RES

        rng = make_rng(seed)
        strm_idx = 0
        task_start = 0
        task_t = tdur + (rng() - 0.5) * 10

        time_minutes = []
        stream_data = []
        desire_data = []
        cumul_avg_desire = []
        is_notification = []
        is_switch_point = []

        last_notif_time = -999
        poll_interval = 60 / freq

        above_thresh_since = None
        total_recall_lag = 0
        recall_count = 0
        notif_precise_count = 0
        notif_total = 0
        desire_sum = 0.0
        switch_count = 0

        for i in range(STEPS + 1):
            t = i * RES
            t_in_task = t - task_start
            d = desire(t_in_task, task_t, k, diff, infl)
            desire_data.append(d)
            desire_sum += d
            cumul_avg_desire.append(desire_sum / (i + 1))

            if d >= THRESHOLD and above_thresh_since is None:
                above_thresh_since = t

            notif_fired = False
            switched = False
            is_notif_time = (t > 0) and (
                abs(t - round(t / poll_interval) * poll_interval) < RES / 2
            )
            if is_notif_time and (t - last_notif_time) >= poll_interval * 0.9:
                notif_fired = True
                last_notif_time = t
                notif_total += 1
                if d >= ACCEPT_FLOOR:
                    notif_precise_count += 1
                if d >= THRESHOLD:
                    strm_idx += 1
                    switch_count += 1
                    switched = True
                    if above_thresh_since is not None:
                        total_recall_lag += t - above_thresh_since
                        recall_count += 1
                    task_start = t
                    task_t = tdur + (rng() - 0.5) * 10
                    above_thresh_since = None

            time_minutes.append(t)
            stream_data.append(strm_idx % 2)
            is_notification.append(notif_fired)
            is_switch_point.append(switched)

        precision = notif_precise_count / notif_total if notif_total > 0 else 0
        avg_recall_lag = total_recall_lag / recall_count if recall_count > 0 else 0
        max_possible_lag = poll_interval
        recall_score = max(0, 1 - avg_recall_lag / (max_possible_lag * 2))
        avg_desire = desire_sum / (STEPS + 1)

        return {
            "time_minutes": time_minutes,
            "stream_data": stream_data,
            "desire_data": desire_data,
            "cumul_avg_desire": cumul_avg_desire,
            "is_notification": is_notification,
            "is_switch": is_switch_point,
            "precision": precision,
            "recall": recall_score,
            "switch_count": switch_count,
            "avg_desire": avg_desire,
            "threshold": THRESHOLD,
            "accept_floor": ACCEPT_FLOOR,
        }

    sim = _run_sim(
        freq_slider.value,
        diff_slider.value,
        steep_slider.value,
        infl_slider.value,
        tdur_slider.value,
        gap_slider.value,
        get_seed(),
    )
    return (sim,)


@app.cell
def _(pl, sim):
    df = pl.DataFrame({
        "time_min": sim["time_minutes"],
        "time_hours": [t / 60.0 for t in sim["time_minutes"]],
        "task_stream": sim["stream_data"],
        "desire": [round(d, 6) for d in sim["desire_data"]],
        "cumul_avg_desire": [round(d, 6) for d in sim["cumul_avg_desire"]],
        "threshold": [sim["threshold"]] * len(sim["time_minutes"]),
        "accept_floor": [sim["accept_floor"]] * len(sim["time_minutes"]),
        "is_notification": sim["is_notification"],
        "is_switch": sim["is_switch"],
    })
    df
    return (df,)


@app.cell
def _(mo, sim):
    mo.hstack(
        [
            mo.stat(
                value=f'{sim["precision"] * 100:.0f}%',
                label="Precision",
                caption="notifs in accept zone",
            ),
            mo.stat(
                value=f'{sim["recall"] * 100:.0f}%',
                label="Recall",
                caption="lag after threshold",
            ),
            mo.stat(
                value=str(sim["switch_count"]),
                label="Context Switches",
                caption="total over 8h",
            ),
            mo.stat(
                value=f'{sim["avg_desire"]:.3f}',
                label="Avg Desire",
                caption="lower is better",
            ),
        ],
        justify="space-around",
    )
    return


@app.cell
def _(df, go):
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["task_stream"].to_list(),
            mode="lines",
            line=dict(color="#534AB7", width=1.5, shape="hv"),
            showlegend=False,
        )
    )
    _fig.update_layout(
        title="Task Stream (0 = focused, 1 = switched)",
        xaxis_title="Time (hours)",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["focused", "switched"],
            range=[-0.1, 1.1],
        ),
        height=220,
        margin=dict(l=80, r=20, t=40, b=40),
    )
    _fig
    return


@app.cell
def _(df, go, pl):
    _notif_df = df.filter(pl.col("is_notification"))

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["desire"].to_list(),
            mode="lines",
            line=dict(color="#534AB7", width=1.5),
            name="desire",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["threshold"].to_list(),
            mode="lines",
            line=dict(color="#D85A30", width=1, dash="dash"),
            name="threshold (0.85)",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["accept_floor"].to_list(),
            mode="lines",
            line=dict(color="#BA7517", width=1, dash="dot"),
            name="accept floor",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_notif_df["time_hours"].to_list(),
            y=_notif_df["desire"].to_list(),
            mode="markers",
            marker=dict(color="rgba(55,138,221,0.7)", size=6),
            name="notification fired",
        )
    )
    _fig.update_layout(
        title="Desire to Switch Over Time",
        xaxis_title="Time (hours)",
        yaxis=dict(range=[0, 1.05], dtick=0.25),
        height=340,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    _fig
    return


@app.cell
def _(df, go):
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["cumul_avg_desire"].to_list(),
            mode="lines",
            line=dict(color="#1D9E75", width=1.5),
            name="cumul. avg desire",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["threshold"].to_list(),
            mode="lines",
            line=dict(color="#D85A30", width=1, dash="dash"),
            name="threshold (0.85)",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=df["time_hours"].to_list(),
            y=df["accept_floor"].to_list(),
            mode="lines",
            line=dict(color="#BA7517", width=1, dash="dot"),
            name="accept floor",
        )
    )
    _fig.update_layout(
        title="Cumulative Avg Desire & Precision/Recall Over Time",
        xaxis_title="Time (hours)",
        yaxis=dict(range=[0, 1.05], dtick=0.25),
        height=340,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Simplified Distraction-Coefficient Model

    We strip away the sigmoid desire-to-switch curve and use a minimal model:

    - Each task has a **natural duration** $T \sim \mathcal{N}(\mu, \sigma^2)$.
    - A linear **desire** $d(t) = \min\!\bigl(1,\; t_{\text{elapsed}} / T\bigr)$
      ramps from 0 at task start to 1 when the task reaches its natural end.
    - **Ground truth**: the user *should* switch once
      $d(t) \ge \tau$ (e.g. $\tau = 0.85$ — task is ≥ 85 % complete).
    - Notifications arrive at a fixed **polling frequency** $f$ (per hour).
    - The **distraction coefficient** $\alpha \in (0, 1]$ controls how
      aggressively the notification nudges the user.  The probability of
      switching at each poll is

    $$
      P(\text{switch}) \;=\; \sigma\!\bigl(\,\beta\,\bigl(d(t) - (1-\alpha)\bigr)\bigr)
    $$

      where $\sigma$ is the logistic sigmoid and $\beta$ controls sharpness.

    | $\alpha$ low | User ignores most notifications | **High precision, low recall** |
    |---|---|---|
    | $\alpha$ high | User switches easily | **Low precision, high recall** |

    By sweeping $\alpha$ at each polling frequency we trace **Pareto frontiers**
    in precision–recall space.

    The **desire curve** $d(t)$ is parametrised as a cubic B-spline through
    editable control points — drag the knots below to reshape it.
    """)
    return


@app.cell
def _():
    import anywidget
    import traitlets

    class SplineEditor(anywidget.AnyWidget):
        _esm = r"""
function render({ model, el }) {
    const W = 540, H = 380;
    const pad = { top: 20, right: 20, bottom: 48, left: 52 };
    const pW = W - pad.left - pad.right, pH = H - pad.top - pad.bottom;
    const R = 8, ns = "http://www.w3.org/2000/svg";

    const wrap = document.createElement("div");
    wrap.style.fontFamily = "system-ui, sans-serif";
    wrap.style.display = "inline-block";

    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    Object.assign(svg.style, {
        background: "#fafafa", border: "1px solid #ddd",
        borderRadius: "6px", cursor: "crosshair", userSelect: "none"
    });
    wrap.appendChild(svg);

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "margin-top:6px;display:flex;gap:8px;align-items:center;";
    function mkBtn(label, fn) {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = "padding:4px 10px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;";
        b.onclick = fn;
        return b;
    }
    btnRow.appendChild(mkBtn("Reset to linear", () => sync([[0,0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1,1]])));
    btnRow.appendChild(mkBtn("Concave (early)", () => sync([[0,0],[0.15,0.45],[0.35,0.75],[0.6,0.92],[1,1]])));
    btnRow.appendChild(mkBtn("Convex (late)", () => sync([[0,0],[0.4,0.08],[0.65,0.25],[0.85,0.55],[1,1]])));
    wrap.appendChild(btnRow);

    const hint = document.createElement("div");
    hint.style.cssText = "font-size:11px;color:#999;margin-top:3px;";
    hint.textContent = "Click to add point \u00b7 drag to move \u00b7 double-click to remove";
    wrap.appendChild(hint);
    el.appendChild(wrap);

    const toX = x => pad.left + x * pW;
    const toY = y => pad.top + (1 - y) * pH;
    const fromX = sx => Math.max(0, Math.min(1, (sx - pad.left) / pW));
    const fromY = sy => Math.max(0, Math.min(1, 1 - (sy - pad.top) / pH));

    function svgEl(tag, a) {
        const e = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(a || {})) e.setAttribute(k, String(v));
        return e;
    }
    function svgTxt(x, y, t, a) {
        const e = svgEl("text", { x, y, "font-size":"11px", fill:"#888", ...a });
        e.textContent = t; return e;
    }

    // static grid
    const sG = svgEl("g"); svg.appendChild(sG);
    sG.appendChild(svgEl("rect", { x:pad.left, y:pad.top, width:pW, height:pH, fill:"#fff", stroke:"#e0e0e0" }));
    for (let v = 0; v <= 1; v += 0.25) {
        sG.appendChild(svgEl("line", { x1:toX(v), y1:pad.top, x2:toX(v), y2:pad.top+pH, stroke:"#f0f0f0" }));
        sG.appendChild(svgEl("line", { x1:pad.left, y1:toY(v), x2:pad.left+pW, y2:toY(v), stroke:"#f0f0f0" }));
        sG.appendChild(svgTxt(toX(v), pad.top+pH+18, v.toFixed(2), { "text-anchor":"middle" }));
        sG.appendChild(svgTxt(pad.left-8, toY(v)+4, v.toFixed(2), { "text-anchor":"end" }));
    }
    sG.appendChild(svgTxt(pad.left+pW/2, H-5, "Task progress (t/T)", { "text-anchor":"middle", fill:"#666", "font-size":"12px" }));
    const yL = svgTxt(14, pad.top+pH/2, "Desire d(t)", { "text-anchor":"middle", fill:"#666", "font-size":"12px" });
    yL.setAttribute("transform", "rotate(-90,14,"+(pad.top+pH/2)+")"); sG.appendChild(yL);
    sG.appendChild(svgEl("line", { x1:toX(0), y1:toY(0), x2:toX(1), y2:toY(1), stroke:"#ddd", "stroke-dasharray":"6,4" }));

    // cubic spline math
    function cSpline(pts) {
        const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]), n = xs.length-1;
        if (n < 1) return null;
        if (n === 1) return { t:"L", xs, ys };
        const h = []; for (let i=0;i<n;i++) h[i]=xs[i+1]-xs[i];
        const al = Array(n+1).fill(0);
        for (let i=1;i<n;i++) { if(!h[i]||!h[i-1]) continue; al[i]=3/h[i]*(ys[i+1]-ys[i])-3/h[i-1]*(ys[i]-ys[i-1]); }
        const l=Array(n+1).fill(0), mu=Array(n+1).fill(0), z=Array(n+1).fill(0); l[0]=1;
        for (let i=1;i<n;i++) { l[i]=2*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1]; if(Math.abs(l[i])<1e-12) l[i]=1e-12; mu[i]=h[i]/l[i]; z[i]=(al[i]-h[i-1]*z[i-1])/l[i]; }
        l[n]=1; const c=Array(n+1).fill(0), b=Array(n).fill(0), d=Array(n).fill(0);
        for (let j=n-1;j>=0;j--) { c[j]=z[j]-mu[j]*c[j+1]; b[j]=(ys[j+1]-ys[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3; d[j]=(c[j+1]-c[j])/(3*h[j]); }
        return { t:"C", a:ys.slice(0,n), b, c:c.slice(0,n), d, xs };
    }
    function evalS(sp, t) {
        if (!sp) return t;
        t = Math.max(sp.xs[0], Math.min(sp.xs[sp.xs.length-1], t));
        if (sp.t==="L") { const r=sp.xs[1]===sp.xs[0]?0:(t-sp.xs[0])/(sp.xs[1]-sp.xs[0]); return sp.ys[0]+r*(sp.ys[1]-sp.ys[0]); }
        let i=0; for(;i<sp.a.length-1;i++) if(t<=sp.xs[i+1]) break;
        const dt=t-sp.xs[i]; return sp.a[i]+sp.b[i]*dt+sp.c[i]*dt*dt+sp.d[i]*dt*dt*dt;
    }

    // dynamic layer
    const dG = svgEl("g"); svg.appendChild(dG);
    let localPts = JSON.parse(model.get("value"));

    function redraw(pts) {
        while (dG.firstChild) dG.removeChild(dG.firstChild);
        const sorted = [...pts].sort((a,b) => a[0]-b[0]);
        const sp = cSpline(sorted);
        let pathD = "";
        for (let i=0; i<=200; i++) {
            const t=i/200, y=Math.max(0,Math.min(1,evalS(sp,t)));
            pathD += (i===0?"M":"L")+toX(t).toFixed(1)+","+toY(y).toFixed(1);
        }
        dG.appendChild(svgEl("path", { d:pathD, fill:"none", stroke:"#534AB7", "stroke-width":2.5 }));
        sorted.forEach((p, i) => {
            const end = i===0 || i===sorted.length-1;
            dG.appendChild(svgEl("circle", {
                cx:toX(p[0]), cy:toY(p[1]), r:R,
                fill: end?"#D85A30":"#534AB7",
                stroke:"#fff", "stroke-width":2, cursor: end?"default":"grab"
            }));
        });
    }

    let timer = null;
    function sync(pts) {
        localPts = [...pts].sort((a,b) => a[0]-b[0]);
        redraw(localPts);
        clearTimeout(timer);
        timer = setTimeout(() => { model.set("value", JSON.stringify(localPts)); model.save_changes(); }, 200);
    }

    let dragIdx = -1;
    function mpos(e) { const r=svg.getBoundingClientRect(); return [e.clientX-r.left, e.clientY-r.top]; }
    function hit(mx, my, pts) {
        for (let i=0; i<pts.length; i++) { const dx=toX(pts[i][0])-mx, dy=toY(pts[i][1])-my; if(dx*dx+dy*dy<(R+4)**2) return i; }
        return -1;
    }

    svg.addEventListener("mousedown", e => {
        const [mx,my] = mpos(e), pts=[...localPts], h=hit(mx,my,pts);
        if (h >= 0) { if (h===0||h===pts.length-1) return; dragIdx=h; svg.style.cursor="grabbing"; e.preventDefault(); return; }
        const nx=fromX(mx), ny=fromY(my);
        if (nx>0.01 && nx<0.99) { pts.push([+nx.toFixed(4),+ny.toFixed(4)]); sync(pts); }
    });
    svg.addEventListener("mousemove", e => {
        if (dragIdx<0) return; e.preventDefault();
        const [mx,my]=mpos(e), pts=[...localPts];
        let nx=fromX(mx), ny=fromY(my);
        const lo=dragIdx>0?pts[dragIdx-1][0]+0.005:0, hi=dragIdx<pts.length-1?pts[dragIdx+1][0]-0.005:1;
        nx=Math.max(lo,Math.min(hi,nx)); ny=Math.max(0,Math.min(1,ny));
        pts[dragIdx]=[+nx.toFixed(4),+ny.toFixed(4)]; sync(pts);
    });
    window.addEventListener("mouseup", () => { if(dragIdx>=0){dragIdx=-1;svg.style.cursor="crosshair";} });
    svg.addEventListener("dblclick", e => {
        const [mx,my]=mpos(e), pts=[...localPts], h=hit(mx,my,pts);
        if (h>0 && h<pts.length-1) { pts.splice(h,1); sync(pts); }
    });

    redraw(localPts);
    model.on("change:value", () => { localPts=JSON.parse(model.get("value")); redraw(localPts); });
}
export default { render };
"""
        value = traitlets.Unicode(
            '[[0,0],[0.25,0.25],[0.5,0.5],[0.75,0.75],[1,1]]'
        ).tag(sync=True)

    return (SplineEditor,)


@app.cell
def _(SplineEditor, mo):
    spline_editor = mo.ui.anywidget(SplineEditor())
    mo.vstack([
        mo.md("### Desire Curve Editor"),
        spline_editor,
    ])
    return (spline_editor,)


@app.cell
def _(json):
    def make_desire_curve(control_points_json):
        """Build a clamped cubic-spline desire function from control-point JSON."""
        pts = (
            json.loads(control_points_json)
            if isinstance(control_points_json, str)
            else list(control_points_json)
        )
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs) - 1

        if n < 1:
            return lambda t: t

        if n == 1:
            def _linear(t):
                t = max(0.0, min(1.0, t))
                span = xs[1] - xs[0]
                if span == 0:
                    return ys[0]
                return ys[0] + (t - xs[0]) / span * (ys[1] - ys[0])
            return _linear

        h = [xs[i + 1] - xs[i] for i in range(n)]
        alpha = [0.0] * (n + 1)
        for i in range(1, n):
            if h[i] == 0 or h[i - 1] == 0:
                continue
            alpha[i] = (
                3 / h[i] * (ys[i + 1] - ys[i])
                - 3 / h[i - 1] * (ys[i] - ys[i - 1])
            )
        l = [0.0] * (n + 1)
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)
        l[0] = 1.0
        for i in range(1, n):
            l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-12:
                l[i] = 1e-12
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n] = 1.0
        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        a_c = ys[:n]

        def _spline(t):
            t = max(xs[0], min(xs[-1], t))
            idx = 0
            for i in range(n):
                if t <= xs[i + 1]:
                    idx = i
                    break
            else:
                idx = n - 1
            dt = t - xs[idx]
            val = a_c[idx] + b[idx] * dt + c[idx] * dt**2 + d[idx] * dt**3
            return max(0.0, min(1.0, val))

        return _spline

    return (make_desire_curve,)


@app.cell
def _(make_desire_curve, spline_editor):
    desire_curve = make_desire_curve(spline_editor.value)
    return (desire_curve,)


@app.cell
def _(desire_curve, math, random):
    def distraction_sim(
        poll_freq,
        alpha,
        task_dur_mean=45.0,
        task_dur_std=12.0,
        day_minutes=480,
        true_threshold=0.85,
        beta=12.0,
        seed=42,
    ):
        """Simulate one 8-hour day with the distraction-coefficient model.

        Returns dict with precision, recall, and TP/FP/FN/TN counts.
        """
        rng = random.Random(seed)
        switch_point = 1.0 - alpha
        poll_interval = 60.0 / poll_freq

        task_start = 0.0
        task_dur = max(10.0, rng.gauss(task_dur_mean, task_dur_std))

        tp = fp = fn = tn = 0

        t = poll_interval
        while t <= day_minutes:
            elapsed = t - task_start
            progress = min(1.0, elapsed / task_dur)
            d = desire_curve(progress)

            should_switch = d >= true_threshold
            p_switch = 1.0 / (1.0 + math.exp(-beta * (d - switch_point)))
            does_switch = rng.random() < p_switch

            if does_switch and should_switch:
                tp += 1
                task_start = t
                task_dur = max(10.0, rng.gauss(task_dur_mean, task_dur_std))
            elif does_switch and not should_switch:
                fp += 1
                task_start = t
                task_dur = max(10.0, rng.gauss(task_dur_mean, task_dur_std))
            elif not does_switch and should_switch:
                fn += 1
            else:
                tn += 1

            t += poll_interval

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return (distraction_sim,)


@app.cell
def _(distraction_sim, go, pl):
    _FREQUENCIES = [2, 4, 8, 12, 20, 30, 60]
    _ALPHAS = [i / 200 for i in range(1, 201)]
    _N_SEEDS = 50

    _rows = []
    for _freq in _FREQUENCIES:
        for _alpha in _ALPHAS:
            _ttp = _tfp = _tfn = _ttn = 0
            for _s in range(_N_SEEDS):
                _res = distraction_sim(_freq, _alpha, seed=_s)
                _ttp += _res["tp"]
                _tfp += _res["fp"]
                _tfn += _res["fn"]
                _ttn += _res["tn"]
            _prec = _ttp / (_ttp + _tfp) if (_ttp + _tfp) > 0 else None
            _rec = _ttp / (_ttp + _tfn) if (_ttp + _tfn) > 0 else None
            if _prec is not None and _rec is not None:
                _rows.append({
                    "freq": _freq,
                    "alpha": round(_alpha, 4),
                    "precision": _prec,
                    "recall": _rec,
                    "tp": _ttp,
                    "fp": _tfp,
                    "fn": _tfn,
                })

    pareto_df = pl.DataFrame(_rows)

    _colors = {
        2: "#E63946",
        4: "#457B9D",
        8: "#2A9D8F",
        12: "#E9C46A",
        20: "#F4A261",
        30: "#264653",
        60: "#6A0572",
    }

    _fig = go.Figure()
    for _freq in _FREQUENCIES:
        _sub = pareto_df.filter(pl.col("freq") == _freq).sort("recall")
        if _sub.height == 0:
            continue
        _fig.add_trace(go.Scatter(
            x=_sub["recall"].to_list(),
            y=_sub["precision"].to_list(),
            mode="lines+markers",
            marker=dict(size=3),
            name=f"{_freq}/hr",
            line=dict(color=_colors[_freq], width=2),
            hovertemplate=(
                "α=%{customdata[0]:.2f}<br>"
                "Recall=%{x:.3f}<br>"
                "Precision=%{y:.3f}<br>"
                "TP=%{customdata[1]}, FP=%{customdata[2]}, FN=%{customdata[3]}"
                "<extra>%{fullData.name}</extra>"
            ),
            customdata=list(zip(
                _sub["alpha"].to_list(),
                _sub["tp"].to_list(),
                _sub["fp"].to_list(),
                _sub["fn"].to_list(),
            )),
        ))

    _fig.update_layout(
        title="Precision–Recall Pareto Frontiers by Polling Frequency<br>"
              "<sub>Each curve sweeps distraction coefficient α from 0→1 "
              "(averaged over 50 seeds, 8-hour day, ~45-min tasks)</sub>",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        height=560,
        margin=dict(l=60, r=20, t=70, b=50),
        legend_title="Poll freq",
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    _fig
    return (pareto_df,)


@app.cell
def _(go, pareto_df, pl):
    _df = pareto_df.with_columns(
        (2 * pl.col("precision") * pl.col("recall")
         / (pl.col("precision") + pl.col("recall"))
        ).alias("f1")
    ).with_columns(pl.col("f1").fill_nan(0.0))

    _best = (
        _df.sort("f1", descending=True)
        .group_by("freq")
        .first()
        .sort("freq")
    )

    _colors = {
        2: "#E63946", 4: "#457B9D", 8: "#2A9D8F",
        12: "#E9C46A", 20: "#F4A261", 30: "#264653", 60: "#6A0572",
    }

    _fig = go.Figure()
    for _row in _best.iter_rows(named=True):
        _sub = _df.filter(pl.col("freq") == _row["freq"]).sort("alpha")
        _fig.add_trace(go.Scatter(
            x=_sub["alpha"].to_list(),
            y=_sub["f1"].to_list(),
            mode="lines",
            name=f'{_row["freq"]}/hr',
            line=dict(color=_colors[_row["freq"]], width=2),
        ))
        _fig.add_trace(go.Scatter(
            x=[_row["alpha"]],
            y=[_row["f1"]],
            mode="markers",
            marker=dict(
                size=10, color=_colors[_row["freq"]],
                symbol="star", line=dict(width=1, color="white"),
            ),
            showlegend=False,
            hovertemplate=(
                f'freq={_row["freq"]}/hr<br>'
                f'best α={_row["alpha"]:.2f}<br>'
                f'F1={_row["f1"]:.3f}<br>'
                f'P={_row["precision"]:.3f}, R={_row["recall"]:.3f}'
                "<extra></extra>"
            ),
        ))

    _fig.update_layout(
        title="F1 Score vs Distraction Coefficient α<br>"
              "<sub>★ = optimal α for each polling frequency</sub>",
        xaxis_title="Distraction coefficient α",
        yaxis_title="F1 Score",
        xaxis=dict(range=[0, 1.0]),
        yaxis=dict(range=[0, 1.05]),
        height=420,
        margin=dict(l=60, r=20, t=70, b=50),
        legend_title="Poll freq",
    )
    _fig
    return


@app.cell
def _(mo, pareto_df, pl):
    _df = pareto_df.with_columns(
        (2 * pl.col("precision") * pl.col("recall")
         / (pl.col("precision") + pl.col("recall"))
        ).alias("f1")
    ).with_columns(pl.col("f1").fill_nan(0.0))

    _best = (
        _df.sort("f1", descending=True)
        .group_by("freq")
        .first()
        .sort("freq")
        .select([
            pl.col("freq").alias("Poll freq (/hr)"),
            pl.col("alpha").alias("Best α"),
            pl.col("precision").round(3).alias("Precision"),
            pl.col("recall").round(3).alias("Recall"),
            pl.col("f1").round(3).alias("F1"),
        ])
    )
    mo.vstack([
        mo.md("### Optimal distraction coefficient per polling frequency"),
        mo.ui.table(_best),
    ])
    return


@app.cell
def _(go, pareto_df, pl):
    _N_SEEDS = 50
    _DAY_MIN = 480

    _df = pareto_df.with_columns([
        ((pl.col("tp") + pl.col("fp")) / _N_SEEDS / (_DAY_MIN / 60)).alias("switches_per_hr"),
        (pl.col("fn") / (pl.col("tp") + pl.col("fn"))).alias("miss_rate"),
    ])

    _colors = {
        2: "#E63946", 4: "#457B9D", 8: "#2A9D8F",
        12: "#E9C46A", 20: "#F4A261", 30: "#264653", 60: "#6A0572",
    }

    _fig = go.Figure()
    for _freq in _df["freq"].unique().sort().to_list():
        _sub = _df.filter(pl.col("freq") == _freq).sort("alpha")
        _fig.add_trace(go.Scatter(
            x=_sub["miss_rate"].to_list(),
            y=_sub["switches_per_hr"].to_list(),
            mode="lines+markers",
            marker=dict(size=3),
            name=f"{_freq}/hr",
            line=dict(color=_colors[_freq], width=2),
            hovertemplate=(
                "α=%{customdata:.2f}<br>"
                "Miss rate=%{x:.2%}<br>"
                "Switches/hr=%{y:.1f}"
                "<extra>%{fullData.name}</extra>"
            ),
            customdata=_sub["alpha"].to_list(),
        ))

    _fig.update_layout(
        title="Practical View: Switches per Hour vs Miss Rate<br>"
              "<sub>Lower-left is better — few interruptions, few missed transitions</sub>",
        xaxis_title="Miss Rate (fraction of should-switch moments missed)",
        yaxis_title="Total Switches per Hour (TP + FP, avg over 50 seeds)",
        xaxis=dict(range=[-0.02, 1.02], tickformat=".0%"),
        height=480,
        margin=dict(l=60, r=20, t=70, b=60),
        legend_title="Poll freq",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Key Takeaways

    1. **Lower polling frequencies dominate.** The 2/hr and 4/hr curves sit
       furthest toward the top-right of the PR plot — they achieve the best
       precision at any given recall level.  This is because fewer polls
       per task mean each notification lands later in the task lifecycle,
       where desire is naturally higher.

    2. **The optimal α decreases as frequency increases.** High-frequency
       polling floods the user with opportunities to switch; you need a
       *less* aggressive notification to avoid drowning signal in noise.
       At 2/hr the optimal α ≈ 0.16, while at 60/hr it drops to ≈ 0.04.

    3. **Diminishing returns on frequency.** Going from 2 → 4/hr barely
       hurts the frontier, but jumping to 30 or 60/hr collapses it — the
       best achievable F1 falls from ~0.87 to ~0.37.

    4. **The "practical view" confirms it.** Plotting switches/hr vs miss
       rate shows that high-frequency polling pushes you into a region of
       many interruptions *and* a still-significant miss rate, whereas
       low-frequency polling can keep interruptions under 2/hr with
       miss rates below 20 %.
    """)
    return


if __name__ == "__main__":
    app.run()
