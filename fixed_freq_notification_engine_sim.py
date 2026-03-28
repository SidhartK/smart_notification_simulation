import marimo

__generated_with = "0.21.1"
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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
