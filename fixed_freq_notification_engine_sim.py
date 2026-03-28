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

    return go, math, mo, pl, random


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
    """)
    return


@app.cell
def _(math, random):
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
            d = min(1.0, elapsed / task_dur)

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


if __name__ == "__main__":
    app.run()
