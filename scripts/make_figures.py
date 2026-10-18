from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

FIG = repo_path("report/figures")
TAB = repo_path("report/tables")
R1, R2, R3 = repo_path("results/task1_dpo"), repo_path("results/task2_ppo"), repo_path("results/task3_grpo")  # results folder per task
R4, R5 = repo_path("results/task4_safety"), repo_path("results/task5_feedback")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots in fixed order
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"  # text, grid and neutral colors
MARKERS = ["o", "s", "^", "D", "v"]  # marker per series

plt.rcParams.update({  # shared figure style
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def jl(path):  # read jsonl or empty list
    return read_jsonl(path) if path.exists() else []


def jload(path):  # read json or none
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name):
    for ext in ("pdf", "png"):  # pdf for the report and png for quick viewing
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("figure", name)


def table(df, name, floatfmt=3):  # write a table as csv, md and tex
    df.to_csv(TAB / f"{name}.csv", index=False)  # csv copy
    cols = list(df.columns)
    fmt = lambda v: f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not math.isnan(v) else ("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")  # markdown copy for pasting into the report
    tex = lambda v: fmt(v).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("β", "$\\beta$").replace("ε", "$\\epsilon$").replace("×", "$\\times$")  # greek letters as math symbols
    body = [" & ".join(tex(c) for c in cols) + " \\\\", "\\midrule"] + [" & ".join(tex(v) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    (TAB / f"{name}.tex").write_text(  # booktabs table scaled to the text width
        "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
    )
    print("table", name)


def smooth(x, w=5):  # moving average for noisy logs
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")  # centered moving average


def small_multiples(series, keys, titles, name, ncols=4, xlabel="update"):  # one panel per logged metric
    rows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(7.0, 1.7 * rows), squeeze=False)
    for ax, key, title in zip(axes.flat, keys, titles):
        for i, (label, log) in enumerate(series.items()):
            x = [r["step"] for r in log]
            ax.plot(x, [r[key] for r in log], color=SERIES[i], marker=MARKERS[i] if len(log) <= 25 else None,
                    markersize=6 - 1.5 * i, label=label)  # shrinking markers keep overlapping runs visible
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)  # hide unused panels
    fig.tight_layout()
    if len(series) > 1:
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(series))  # legend below so no line is covered
    save(fig, name)


def eval_row(name, e, extra=None):  # one row of held out eval metrics
    return {
        "run": name, "n": e["eval_examples"], "reward": e["reward_mean"], "reward_se": e.get("reward_se", e["reward_std"] / math.sqrt(e["eval_examples"])),
        "kl_per_token": e["kl_per_token"], "kl_per_seq": e["kl_per_sequence"], "entropy_per_token": e.get("entropy_per_token"),
        "length_mean": e["length_mean"], "length_std": e["length_std"], "length_iqr": e["length_iqr"],
        "eos_rate": e.get("eos_rate"), "truncation_rate": e["truncation_rate"], **(extra or {}),
    }


def task1():  # dpo figures and tables
    runs = {"base": "base (no training)", "standard": "standard (1446 pairs)", "beta_0.03": "β=0.03 (600)", "beta_0.1": "β=0.10 (600)", "beta_0.3": "β=0.30 (600)"}
    rows = []
    for k, label in runs.items():  # summary row per dpo run
        e = jload(R1 / f"{k}_eval.json")
        n = e["eval_examples"]
        rows.append({"run": label, "beta": e["beta"], "dpo_loss": e["dpo_loss"], "pref_acc": e["preference_accuracy"],
                     "pref_acc_se": math.sqrt(e["preference_accuracy"] * (1 - e["preference_accuracy"]) / n),
                     "ref_acc": e["reference_preference_accuracy"], "margin": e["log_ratio_margin_mean"],
                     "kl_per_seq": e["kl_per_sequence"], "reward": e["reward_mean"], "reward_se": e["reward_std"] / math.sqrt(n),
                     "length_mean": e["length_mean"], "length_std": e["length_std"], "truncation": e["truncation_rate"]})
    table(pd.DataFrame(rows), "t1_dpo_summary")

    logs = {"standard": jl(R1 / "standard_train_log.jsonl"), "β=0.03": jl(R1 / "beta_0.03_train_log.jsonl"), "β=0.10": jl(R1 / "beta_0.1_train_log.jsonl"),  # training logs per run
            "β=0.30": jl(R1 / "beta_0.3_train_log.jsonl"), "length balanced": jl(R1 / "length_balanced_train_log.jsonl")}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for i, (label, log) in enumerate(logs.items()):
        x = [r["step"] for r in log]
        axes[0].plot(x, smooth([r["loss"] for r in log]), color=SERIES[i], label=label)
        axes[1].plot(x, smooth([r["preference_accuracy"] for r in log]), color=SERIES[i])
    std = logs["standard"]
    x = [r["step"] for r in std]
    axes[2].plot(x, smooth([r["chosen_reward"] for r in std]), color=SERIES[0], label="chosen")
    axes[2].plot(x, smooth([r["rejected_reward"] for r in std]), color=SERIES[1], label="rejected")
    axes[0].axhline(math.log(2), color=NEUTRAL, ls="--", lw=1)  # log 2 is the loss before any update
    axes[0].set_title("training loss (5-step avg)"); axes[1].set_title("batch reward accuracy"); axes[2].set_title("implicit reward, standard run")
    for ax in axes:
        ax.set_xlabel("optimizer step")
    axes[0].legend(fontsize=6); axes[2].legend()
    fig.tight_layout()
    save(fig, "f1_dpo_training")

    forks = [0.03, 0.10, 0.30]  # beta study figure
    fr = [r for r in rows if r["run"].startswith("β")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    panels = [("pref_acc", "pref_acc_se", "held-out preference accuracy"), ("kl_per_seq", None, "KL to reference per sequence"), ("reward", "reward_se", "reward model score")]
    for ax, (key, se, title) in zip(axes, panels):
        ax.errorbar(range(3), [r[key] for r in fr], yerr=[r[se] for r in fr] if se else None, color=SERIES[0], marker="o", capsize=3, label="600-pair forks")
        ax.axhline(rows[1][key], color=SERIES[1], ls="--", lw=1, label="standard (full epoch)")  # different budget shown as a reference line
        ax.set_xticks(range(3)); ax.set_xticklabels(["0.03", "0.10", "0.30"]); ax.set_xlim(-0.3, 2.3); ax.set_xlabel("β"); ax.set_title(title)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "f1_beta_study")

    la = jload(R1 / "length_analysis.json")  # length strata results
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    srows = []
    for model, v in la.items():
        for s in strata:
            x = v["per_stratum"][s]
            srows.append({"model": model, "stratum": s, "n": x["n"], "pref_acc": x["preference_accuracy"],
                          "pref_acc_se": math.sqrt(x["preference_accuracy"] * (1 - x["preference_accuracy"]) / x["n"]),
                          "margin": x["log_ratio_margin_mean"], "dpo_loss": x["dpo_loss"], "ref_acc": x["reference_preference_accuracy"]})
    sdf = pd.DataFrame(srows)
    table(sdf, "t1_length_strata")
    rq = jload(R1 / "rq_support.json")
    comp = rq["rq2_dataset_length"] if rq else None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.4, 1]})  # accuracy by stratum and training mix
    w = 0.38
    for i, model in enumerate(["standard", "length_balanced"]):
        d = sdf[sdf["model"] == model]
        xs = np.arange(3) + (i - 0.5) * w
        axes[0].bar(xs, d["pref_acc"], width=w - 0.03, yerr=d["pref_acc_se"], color=SERIES[i], capsize=2, label=model.replace("_", " "))
    axes[0].axhline(0.5, color=NEUTRAL, ls="--", lw=1)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(["preferred longer", "matched", "rejected longer"])
    axes[0].set_title("held-out accuracy by length stratum"); axes[0].legend(fontsize=6); axes[0].grid(axis="x", visible=False)
    if comp:  # training pair composition panel
        names = [("standard_train", "standard train"), ("length_balanced_train", "balanced train")]
        left = np.zeros(2)
        for j, (key, lab) in enumerate([("preferred_longer_share", "preferred longer"), ("length_matched_share", "matched"), ("rejected_longer_share", "rejected longer")]):
            vals = np.array([comp[n][key] for n, _ in names])
            axes[1].barh(range(2), vals, left=left, color=SERIES[j + 2] if j < 2 else SERIES[4], edgecolor="white", linewidth=1.5, label=lab)
            for y, (v, l) in enumerate(zip(vals, left)):
                axes[1].text(l + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)  # share printed on each segment
            left += vals
        axes[1].set_yticks(range(2)); axes[1].set_yticklabels([lab for _, lab in names]); axes[1].set_xlim(0, 1)
        axes[1].set_title("training pair composition"); axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3); axes[1].grid(axis="y", visible=False)
        table(pd.DataFrame([{"dataset": n, **comp[n]} for n in ["standard_train", "standard_eval", "length_balanced_train"]]), "t1_dataset_length")
    fig.tight_layout()
    save(fig, "f1_length_strata")

    wl = []  # word limit table
    base = jload(R1 / "word_limit_base.json")
    entries = ([("base", base["summary"])] if base else []) + [(m, la[m]["word_limit"]) for m in ["standard", "length_balanced"]]
    for m, s in entries:
        wl.append({"model": m, "compliance": s["compliance_rate"], "words_mean": s["words_mean"], "words_std": s["words_std"],
                   "tokens_mean": s["length_tokens_mean"], "tokens_std": s["length_tokens_std"], "overshoot_ratio": s["mean_overshoot_ratio"]})
    table(pd.DataFrame(wl), "t1_word_limit")


def rl_task(rdir, prefix, tprefix, runs, traj_keys, traj_titles):  # shared ppo and grpo outputs
    std_log = jl(rdir / "standard_log.jsonl")
    small_multiples({"standard": std_log}, traj_keys, traj_titles, f"{prefix}_standard_trajectories")  # standard run trajectories
    s = jload(rdir / "standard_summary.json")
    keep = {k: v for k, v in s.items() if k not in ("config", "output")}  # summary without the config dump
    table(pd.DataFrame([keep]), f"{tprefix}_standard_summary")
    table(pd.DataFrame([eval_row(label, jload(rdir / f"{k}_eval.json")) for k, label in runs.items() if (rdir / f"{k}_eval.json").exists()]), f"{tprefix}_eval_runs")
    return std_log


def task2():  # ppo figures and tables
    runs = {"midpoint": "midpoint", "standard": "standard (20)", "fork_eps0.05_kl0.1": "ε=0.05", "fork_eps0.2_kl0.1": "ε=0.20, βKL=0.10 (default)",
            "fork_eps0.5_kl0.1": "ε=0.50", "fork_eps0.2_kl0.0": "βKL=0.0", "fork_eps0.2_kl0.2": "βKL=0.20"}
    rl_task(R2, "f2_ppo", "t2_ppo", runs,
            ["reward", "kl", "policy_loss", "value_loss", "entropy", "clip_fraction", "policy_grad_norm", "response_length"],
            ["reward", "KL to reference", "policy loss", "value loss", "entropy", "clip fraction", "policy grad norm", "response length"])
    clip = jload(R2 / "clipping_study.json")  # clipping study
    cb = clip["cached_batch"]["per_epsilon"]
    eps = list(cb)
    table(pd.DataFrame([{"epsilon": float(e), **cb[e]} for e in eps]), "t2_clipping_cached")
    fig, ax = plt.subplots(figsize=(3.4, 2.1))  # cached batch clip bars
    xs = np.arange(len(eps)); w = 0.38
    for i, (key, lab) in enumerate([("clip_fraction", "ratio outside [1-ε, 1+ε]"), ("affected_token_fraction", "clipping changes the objective")]):
        vals = [cb[e][key] for e in eps]
        ax.bar(xs + (i - 0.5) * w, vals, width=w - 0.03, color=SERIES[i], label=lab)
        for x, v in zip(xs + (i - 0.5) * w, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6)  # value above each bar
    ax.set_xticks(xs); ax.set_xticklabels([f"ε={float(e):.2f}" for e in eps]); ax.set_ylabel("fraction of tokens")
    ax.set_title("cached batch (8,814 tokens)"); ax.legend(fontsize=6); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "f2_clipping_cached")
    table(pd.DataFrame([{"epsilon": e, "policy_grad_norm_mean": f["stability"]["policy_grad_norm"]["mean"], "policy_loss_std": f["stability"]["policy_loss"]["std"],  # epsilon fork table
                         "clip_fraction_mean": f["stability"]["clip_fraction"]["mean"], "ratio_abs_dev_mean": f["stability"]["ratio_abs_dev"]["mean"],
                         "final_kl": f["stability"]["final_kl"], "eval_reward": f["eval"]["reward_mean"]} for e, f in clip["forks"].items()]), "t2_clipping_forks")

    kl = jload(R2 / "kl_study.json")  # kl study
    logs = {f"βKL={float(k):.2f}": jl(R2 / f"{f['run']}_log.jsonl") for k, f in kl["forks"].items()}
    small_multiples(logs, ["reward", "kl", "entropy", "response_length"], ["reward", "KL to reference", "entropy", "response length"], "f2_kl_trajectories")
    rq = jload(R2 / "rq_support.json")  # rq tables
    if rq:
        fc = rq["rq2_kl_first_change"]
        table(pd.DataFrame([{"comparison": c, **{k: fc[c][k]["first_update_differs"] for k in ["policy_loss", "kl", "entropy", "reward", "response_length", "clip_fraction"]},
                             "text": fc[c]["response_text_first_differs"]} for c in fc]), "t2_kl_first_change")
        ri = rq["rq3_reward_informativeness"]
        table(pd.DataFrame([{"run": r, **{k: ri[r][k] for k in ["corr_reward_length", "corr_reward_length_finished", "corr_reward_seq_kl", "reward_eos", "reward_no_eos"]}}
                            for r in ri if r != "standard_vs_midpoint_paired"]), "t2_reward_informativeness")


def task3():  # grpo figures and tables
    runs = {"midpoint": "midpoint", "standard": "standard (20)", "fork_grpo": "grpo fork (8)", "fork_dr_grpo": "dr_grpo fork (8)"}
    rl_task(R3, "f3_grpo", "t3_grpo", runs,
            ["reward", "reward_std", "kl", "entropy", "loss", "grad_norm", "response_length", "masked_completions"],
            ["mean group reward", "within-group reward std", "KL to reference", "entropy", "loss", "grad norm", "mean response length", "masked completions (of 4)"])
    gs = jload(R3 / "group_size_study.json")  # group size study
    ks = [int(k) for k in gs["per_k"]]
    rows = [{"K": k, "bin": b, **gs["per_k"][str(k)][b]} for k in ks for b in ["all", "hard", "easy"]]
    table(pd.DataFrame(rows), "t3_group_size")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for i, b in enumerate(["all", "hard", "easy"]):
        for ax, key in zip(axes, ["informative_rate", "mean_within_group_std", "var_centered_reward"]):
            ax.plot(ks, [gs["per_k"][str(k)][b][key] for k in ks], color=SERIES[i], marker="o", label=b)
    for ax, t in zip(axes, ["informative-group rate", "mean within-group reward std", "variance of centered reward"]):
        ax.set_xticks(ks); ax.set_xlabel("K (192 generations fixed)"); ax.set_title(t)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "f3_group_size")

    norm = jload(R3 / "normalization_study.json")  # normalization study
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = []
    for i, (lt, f) in enumerate(norm["forks"].items()):
        log = jl(R3 / f"{f['run']}_log.jsonl")
        pts = [(n, w) for r in log for n, w, t in zip(r["lengths"], r["seq_weight"], r["truncated"]) if not t]  # kept completions only
        axes[0].scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=SERIES[i], edgecolor="white", linewidth=0.6, label=lt)
        lc = f["length_conditioned"]
        rows.append({"loss_type": lt, **lc, **{f"eval_{k}": f["eval"][k] for k in ["reward_mean", "reward_se", "kl_per_sequence", "length_mean", "eos_rate"]}})
    axes[0].set_xlabel("completion length (tokens)"); axes[0].set_ylabel("|A| × tokens / normalizer"); axes[0].set_title("per-completion gradient weight"); axes[0].legend(fontsize=6)
    lts = [r["loss_type"] for r in rows]  # weight share bars
    xs = np.arange(len(lts)); w = 0.38
    for j, (key, lab) in enumerate([("weight_share_short", "short half"), ("weight_share_long", "long half")]):
        vals = [r[key] for r in rows]
        axes[1].bar(xs + (j - 0.5) * w, vals, width=w - 0.03, color=SERIES[j + 2], label=lab)
        for x, v in zip(xs + (j - 0.5) * w, vals):
            axes[1].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(lts); axes[1].set_title("share of gradient weight"); axes[1].legend(fontsize=6); axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "f3_normalization")
    table(pd.DataFrame(rows), "t3_normalization")
    rq = jload(R3 / "rq_support.json")  # instability table
    if rq:
        table(pd.DataFrame([{"run": k, **v} for k, v in rq["rq3_instability"].items()]), "t3_instability")


def task4():  # safety figures and tables
    s = jload(R4 / "safety_summary.json")
    if not s:
        return
    pol = list(s["policies"])
    keys = ["safe_answer_rate", "safe_over_refusal_rate", "safe_any_refusal_rate", "unsafe_compliance_rate", "unsafe_any_answer_rate",
            "unsafe_justified_refusal_rate", "ambiguous_rate", "judge_class_mismatch_rate", "mean_response_tokens", "std_response_tokens"]
    table(pd.DataFrame([{"policy": p, **{k: s["policies"][p][k] for k in keys}} for p in pol]), "t4_safety_rates")
    metrics = [("safe_answer_rate", "safe: answered"), ("safe_any_refusal_rate", "safe: refused (any)"),  # rate bars per policy
               ("unsafe_justified_refusal_rate", "unsafe: refused"), ("unsafe_any_answer_rate", "unsafe: answered (any)")]
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    xs = np.arange(len(metrics)); w = 0.2
    for i, p in enumerate(pol):
        vals = [s["policies"][p][k] for k, _ in metrics]
        ax.bar(xs + (i - 1.5) * w, vals, width=w - 0.02, color=SERIES[i], label=p.upper())
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in metrics]); ax.set_ylabel("rate (AI judge)"); ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=6.5, loc="upper center"); ax.grid(axis="x", visible=False); ax.set_title("XSTest behavior by policy (rates by prompt class)")
    fig.tight_layout()
    save(fig, "f4_safety_rates")

    cat = pd.read_csv(R4 / "category_labels.csv")  # label counts per category
    safe = cat[cat["benchmark_class"] == "SAFE"].copy()
    safe["refusal"] = (safe["JUSTIFIED_REFUSAL"] + safe["OVER_REFUSAL"]) / safe[["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]].sum(axis=1)
    piv = safe.pivot(index="type", columns="policy", values="refusal")[pol]
    table(piv.reset_index(), "t4_safe_category_refusal")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))  # refusal heatmap
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")  # single hue light to dark
    ax.set_xticks(range(len(pol))); ax.set_xticklabels([p.upper() for p in pol]); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6.5)
    for y in range(piv.shape[0]):
        for x in range(piv.shape[1]):
            v = piv.values[y, x]
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.55 else INK)  # value in each cell
    ax.grid(False); ax.set_title("refusal rate on safe prompts by XSTest category")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    save(fig, "f4_safe_category_refusal")
    labs = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]  # full label mix per category
    full = cat.copy()
    full["n"] = full[labs].sum(axis=1)
    for l in labs:
        full[l] = full[l] / full["n"]  # label share within each category and policy
    full["policy"] = pd.Categorical(full["policy"], pol, ordered=True)
    full["benchmark_class"] = pd.Categorical(full["benchmark_class"], ["SAFE", "UNSAFE"], ordered=True)
    full = full.sort_values(["policy", "benchmark_class", "type"]).reset_index(drop=True)
    table(full[["policy", "benchmark_class", "type", "n"] + labs], "t4_category_labels")
    types = list(full[full["policy"] == pol[0]]["type"])
    lab_col = [SERIES[0], SERIES[2], SERIES[1], SERIES[3], NEUTRAL]
    fig, axes = plt.subplots(1, len(pol), figsize=(7.0, 3.5), sharey=True)  # one stacked panel per policy
    for ax, p_ in zip(axes, pol):
        d = full[full["policy"] == p_].set_index("type").loc[types]
        left = np.zeros(len(types))
        for l, col in zip(labs, lab_col):
            ax.barh(range(len(types)), d[l].values, left=left, color=col, edgecolor="white", linewidth=0.8, label=l.replace("_", " ").lower())
            left += d[l].values
        n_safe = int((d["benchmark_class"] == "SAFE").sum())
        ax.axhline(n_safe - 0.5, color=INK2, linewidth=0.8)  # safe categories above the line and unsafe contrasts below
        ax.set_xlim(0, 1); ax.set_title(p_.upper()); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(types))); axes[0].set_yticklabels(types, fontsize=6); axes[0].invert_yaxis()
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5)
    fig.tight_layout()
    save(fig, "f4_category_labels")
    a = s.get("manual_audit")  # manual audit tables
    if a and a.get("labeled_rows"):
        table(pd.DataFrame([{"labeled_rows": a["labeled_rows"], "agreement": a["agreement"], "cohen_kappa": a["cohen_kappa"],
                             "ai_ambiguous": a["ai_ambiguous_rate_on_audit"], "manual_ambiguous": a["manual_ambiguous_rate"]}]), "t4_audit_agreement")
        table(pd.DataFrame(a["confusion_manual_rows_ai_cols"]).T.reset_index().rename(columns={"index": "manual \\ ai"}), "t4_audit_confusion", floatfmt=0)


def task5():  # math feedback figures and tables
    comp = jload(R5 / "feedback_comparison.json")
    gsm, tr = jload(R5 / "gsm_eval.json"), jload(R5 / "transfer_eval.json")
    if not comp:
        return
    rows = []
    for ds, d in [("GSM8K", gsm), ("SVAMP", tr)]:  # one row per dataset and policy
        for p, e in d["policies"].items():
            n = d["examples"]
            rows.append({"dataset": ds, "policy": p, "exact_acc": e["exact_accuracy"], "acc_se": math.sqrt(e["exact_accuracy"] * (1 - e["exact_accuracy"]) / n),
                         "format": e["format_compliance"], "tokens": e["mean_response_tokens"], **e["failure_types"],
                         "win_vs_sft": e.get("win_rate_vs_sft"), "wins": e.get("wins"), "losses": e.get("losses"), "ties": e.get("ties"),
                         "verifier_judge_agreement": e.get("verifier_judge_agreement"),
                         "judge_decisive_on_verifier_ties": e.get("judge_decisive_when_verifier_ties"), "verifier_ties": e.get("verifier_tie_pairs")})
    df = pd.DataFrame(rows)
    gsm_ref = df[df["dataset"] == "GSM8K"].set_index("policy")  # in-domain value of the same policy
    is_tr = df["dataset"] == "SVAMP"
    df.insert(df.columns.get_loc("acc_se") + 1, "acc_drop_vs_gsm8k",
              [gsm_ref.loc[p_, "exact_acc"] - a if t else np.nan for p_, a, t in zip(df["policy"], df["exact_acc"], is_tr)])
    df.insert(df.columns.get_loc("win_vs_sft") + 1, "win_drop_vs_gsm8k",
              [gsm_ref.loc[p_, "win_vs_sft"] - w if t and p_ != "sft" else np.nan for p_, w, t in zip(df["policy"], df["win_vs_sft"], is_tr)])
    table(df, "t5_math_eval")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))  # accuracy and failure type figure
    pols = ["sft", "rlvr", "rlaif"]
    xs = np.arange(2); w = 0.26
    for i, p in enumerate(pols):
        d = df[df["policy"] == p]
        axes[0].bar(xs + (i - 1) * w, d["exact_acc"], width=w - 0.02, yerr=d["acc_se"], capsize=2, color=SERIES[i], label=p.upper())
    axes[0].set_xticks(xs); axes[0].set_xticklabels(["GSM8K (300)", "SVAMP (100)"]); axes[0].set_title("exact final-answer accuracy"); axes[0].legend(fontsize=6); axes[0].grid(axis="x", visible=False)
    labels = [f"{p.upper()}\n{ds}" for ds in ["GSM8K", "SVAMP"] for p in pols]  # failure type bars
    left = np.zeros(len(labels))
    for j, (key, lab, col) in enumerate([("wrong_number", "wrong number", SERIES[1]), ("no_final_answer", "no #### answer", SERIES[3]), ("cut_off_at_limit", "cut off", NEUTRAL)]):
        vals = np.array([df[(df["policy"] == p) & (df["dataset"] == ds)][key].iloc[0] / (300 if ds == "GSM8K" else 100) for ds in ["GSM8K", "SVAMP"] for p in pols])
        axes[1].bar(range(len(labels)), vals, bottom=left, color=col, edgecolor="white", linewidth=1.2, label=lab)
        left += vals
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=6); axes[1].set_title("failure types (share of all problems)"); axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3); axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "f5_math_accuracy")

    diag = comp["diagnostic"]["per_variant"]  # perturbation diagnostics
    variants = list(diag)
    short = {"corrupt_reasoning_correct_final": "corrupt reasoning,\ncorrect final", "good_reasoning_wrong_final": "good reasoning,\nwrong final",
             "persuasive_filler_correct": "persuasive filler", "gold_distractor_wrong_final": "gold distractor,\nwrong final"}
    table(pd.DataFrame([{"variant": v, "mechanism": m, **diag[v][m]} for v in variants for m in ["verifier", "judge"]]), "t5_diagnostics")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)  # verifier and judge panels
    for ax, mech in zip(axes, ["verifier", "judge"]):
        left = np.zeros(len(variants))
        for key, lab, col in [("better_rate", "prefers clean", SERIES[0]), ("tie_rate", "tie", NEUTRAL), ("wrong_preference_rate", "prefers perturbed", SERIES[1])]:
            vals = np.array([diag[v][mech][key] for v in variants])
            ax.barh(range(len(variants)), vals, left=left, color=col, edgecolor="white", linewidth=1.2, label=lab)
            left += vals
        ax.set_xlim(0, 1); ax.set_title("exact verifier" if mech == "verifier" else "pairwise AI judge"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(variants))); axes[0].set_yticklabels([short[v] for v in variants], fontsize=6.5); axes[0].invert_yaxis()
    fig.tight_layout()
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=6.5)
    save(fig, "f5_diagnostics")
    d = comp["diagnostic"]
    table(pd.DataFrame([{"score": k, "verifier": d[k]["verifier"], "judge": d[k]["judge"]} for k in ["S_reason", "S_outcome", "S_outcome_incl_distractor"]]), "t5_sensitivity")
    audit = jload(R5 / "judge_tie_audit.json")
    if audit:  # real ties versus multi label outputs and first shown picks
        table(pd.DataFrame([{"set": k, **{c: v[c] for c in ["pairs", "a_wins", "b_wins", "true_ties", "unparseable", "multi_label", "picked_first_shown", "matches_cached_result"]}}
                            for k, v in audit.items()]), "t5_judge_audit", floatfmt=0)


def setup_tables():  # setup table per task
    dpo, ppo, grpo, fb = (load_yaml(f"configs/{n}.yaml") for n in ["dpo", "ppo", "grpo", "feedback"])
    lora = dpo["lora"]
    kept = lambda name: jload(R1 / f"filtered_{name}.json")  # rows kept after the prompt length rule

    def counts(name, total):
        f = kept(name)
        return f"{f['kept']} of {total} ({f['dropped']} dropped)" if f else str(total)

    rows1 = [  # task 1 dpo
        ("policy", f"{dpo['base_model']} + LoRA (r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']}, {', '.join(lora['target_modules'])})"),
        ("reference", "same base model, adapter disabled"),
        ("optimizer", f"AdamW, lr {dpo['learning_rate']}, weight decay {dpo['weight_decay']}, grad clip {dpo['max_grad_norm']}"),
        ("batch", f"{dpo['batch_size']} pairs x {dpo['grad_accum_steps']} accumulation = {dpo['batch_size'] * dpo['grad_accum_steps']} pairs per step"),
        ("standard run", f"beta {dpo['beta']}, {dpo['epochs']} epoch, {counts('dpo_standard_train', 1500)} pairs"),
        ("beta forks", f"beta in {dpo['betas']}, first {dpo['short_ablation_examples']} filtered pairs, same init and seed"),
        ("length balanced run", f"{counts('dpo_length_balanced_train', 1500)} pairs, 1 epoch"),
        ("held out pairs", f"standard {counts('standard_eval', 300)}; strata {counts('length_eval', 246)}"),
        ("sequence rule", f"max {dpo['max_sequence_length']} tokens, prompt kept, response cut from the right"),
        ("generation", f"temperature {fb['generation']['temperature']}, top-p {fb['generation']['top_p']}, {dpo['max_generation_tokens']} new tokens"),
        ("reward model", dpo["reward_model"]),
        ("word limit prompts", "10 fixed prompts"),
        ("seed", str(dpo["seed"])),
    ]
    rows2 = [  # task 2 ppo
        ("start", "supplied PPO midpoint policy LoRA and value model"),
        ("value model", f"{ppo['value_model_init']} with LoRA + scalar head, trainable weights in fp32"),
        ("updates", f"standard {ppo['updates']}, forks {ppo['fork_updates']}, {ppo['prompts_per_update']} prompt per update, {ppo['ppo_epochs']} PPO epochs"),
        ("learning rates", f"policy {ppo['policy_learning_rate']}, value LoRA {ppo['value_lora_learning_rate']}, value head {ppo['value_head_learning_rate']}"),
        ("clipping", f"epsilon {ppo['clip_epsilon']}; study {ppo['clip_values']}"),
        ("KL penalty", f"beta_KL {ppo['kl_beta']}; study {ppo['kl_values']}"),
        ("GAE", f"gamma {ppo['gamma']}, lambda {ppo['gae_lambda']}; value coef {ppo['value_coef']}"),
        ("missing EOS penalty", str(ppo["missing_eos_penalty"])),
        ("lengths", f"prompt {ppo['max_prompt_length']}, train response {ppo['max_response_length']}, eval response {ppo['eval_max_response_length']}, reward input {ppo['reward_max_length']}"),
        ("grad clip", str(ppo["max_grad_norm"])),
        ("prompts", "1200 train pool (seeded shuffle, update k uses k-th prompt), 200 eval"),
        ("cached batch", "32 supplied rollouts, 8814 response tokens"),
    ]
    rows3 = [  # task 3 grpo
        ("start", "supplied GRPO midpoint policy LoRA"),
        ("updates", f"standard {grpo['updates']}, forks {grpo['fork_updates']}, {grpo['prompts_per_update']} prompt per update, {grpo['policy_epochs']} policy epoch"),
        ("group", f"K = {grpo['num_generations']} completions per prompt, advantages normalized within the group"),
        ("optimizer", f"AdamW, lr {grpo['learning_rate']}, grad clip {grpo['max_grad_norm']}"),
        ("objective", f"clipped ratio, epsilon {grpo['clip_epsilon']}, KL penalty beta {grpo['kl_beta']} (k3 estimator)"),
        ("lengths", f"prompt {grpo['max_prompt_length']}, completion {grpo['max_completion_length']}, completions at the cap masked, eval {grpo['eval_max_response_length']}"),
        ("normalization study", "grpo (divide by own length) vs dr_grpo (divide by 512), same prompts, seed, budget"),
        ("group size study", f"K in {grpo['group_sizes']} from the supplied cache, 24 prompts x 8 = 192 completions"),
        ("prompts", "1200 train pool (seeded shuffle), 200 eval"),
    ]
    rows4 = [  # task 4 safety
        ("policies", "SFT (base, no adapter), standard DPO, standard PPO, standard GRPO"),
        ("prompts", "XSTest, 450 (250 safe, 200 unsafe), 10 safe and 8 unsafe categories"),
        ("decoding", f"greedy, {fb['safety_max_new_tokens']} new tokens"),
        ("AI judge", f"{fb['ai_judge_model']}, greedy, {fb['judge_max_new_tokens']} tokens, 5 labels"),
        ("manual audit", f"240 responses ({2 * fb['manual_audit_per_class']} prompts x 4 policies; {fb['manual_audit_per_class']} safe, {fb['manual_audit_per_class']} unsafe, seed {fb['seed']}), manual human reviewed blind to AI label and prompt class"),
    ]
    rows5 = [  # task 5 feedback
        ("policies", "SFT (base), RLVR and RLAIF (supplied adapters)"),
        ("data", "GSM8K eval 300, SVAMP transfer 100, controlled diagnostics 100 (20 problems x 5 variants)"),
        ("decoding", f"greedy, {fb['math_max_new_tokens']} new tokens"),
        ("verifier", "exact match of the number after the last #### marker"),
        ("AI judge", f"{fb['ai_judge_model']}, pairwise, greedy, 4 tokens, A/B order set by hash"),
        ("win rate", "win 1, tie 0.5, loss 0, each policy vs SFT on the same problem"),
    ]
    for name, rows in [("t1_setup", rows1), ("t2_setup", rows2), ("t3_setup", rows3), ("t4_setup", rows4), ("t5_setup", rows5)]:
        table(pd.DataFrame(rows, columns=["setting", "value"]), name)


def main():  # build every figure and table
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for step in (setup_tables, task1, task2, task3, task4, task5):  # each task draws only what exists
        step()


if __name__ == "__main__":
    main()
