#!/usr/bin/env python3
"""
build_paper.py -- assembles paper_make.tex from paper_src/*.tex and the saved result files.

Every table that reports a result of this study is generated here from a saved result file
(budget_ladder.json, runs/frontier_v1*/frontier_rows.csv, runs/common_kl_comparison_v4_samples/*,
...) and never typed by hand; the older appendix tables (full per-model comparisons) are kept as
LaTeX blocks and are checked against the saved results by verify_paper_tables.py.

Usage:  python3 build_paper.py [--out paper_make.tex]
"""
import argparse, csv, glob, json, math, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "paper_src")


def rd(name):
    return open(os.path.join(SRC, name)).read()


# ----------------------------------------------------------------------------- tables
def fx(v, nd=2):
    return f"{v:.{nd}f}"


def tab_ladder(d):
    base = d["systems"][0]
    p = base["p"]
    keep = [0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    rows = []
    for r in base["grid"]:
        if round(r["eps"], 4) not in keep:
            continue
        R = r["rules"]
        a, rl, s, lm = R["abs"], R["rel"], R["sched"], R["lam"]
        rows.append(
            f"{r['eps']:.2f} & {r['d_ber']:.3f} & {a['ratio']:.2f} & {rl['ratio']:.2f} & {s['ratio']:.2f} & "
            f"{lm['ratio']:.3f} & {100*a['infeasible_mass']:.0f}\\% & {100*a['kl_share_infeasible']:.0f}\\% & "
            f"{1+a['cv2_lambda']:.1f} / {1+rl['cv2_lambda']:.1f} \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Base system ($R{=}1$, $N{=}6$, reference loop probability $p=%.4f$): KL spent as a multiple of the minimum $d_{\mathrm{Ber}}(\epsilon\Vert p)$, for each rule of the ladder at the same exact loop probability $\epsilon$. FB: fixed budget; AF: always-feasible budget; schedule: best history-independent budget schedule; FP: fixed price (one multiplier). The optimum is 1 by construction. ``Infeasible'': share of node visits at which the FB budget is below the achievable floor, and the share of the FB controller's KL spent there. $1+\mathrm{CV}^2$ is the second-order waste factor of Proposition~\ref{prop:waste} for FB / AF. All quantities are exact.}
\label{tab:ladder}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}rrrrrrrrr@{}}
\toprule
$\epsilon$ & min KL (bits) & FB & AF & schedule & FP & infeasible visits & KL at infeasible & $1+\mathrm{CV}^2$ (FB / AF) \\
\midrule
""" % p + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}""")


def tab_smalld(d):
    rows = []
    for c in d["small_distortion"]:
        a, rl, lm = c["abs"], c["rel"], c["lam"]
        rows.append(f"{c['D']:g} & {a['exact_ratio']:.1f} & {a['R_pred']:.1f} & {a['R_lower_1_over_phi']:.1f} & "
                    f"{rl['exact_ratio']:.2f} & {rl['R_pred']:.2f} & {lm['exact_ratio']:.4f} \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Small reductions in the base system. Exact KL as a multiple of the minimum, against the second-order prediction $1+\mathrm{CV}^2$ and, for the fixed budget, the lower bound $1/\varphi$ of Proposition~\ref{prop:waste}. $D=p-\epsilon$. FB: fixed budget; AF: always-feasible budget; FP: fixed price.}
\label{tab:smalld}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{FB} & \multicolumn{2}{c}{AF} & FP \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}
$D$ & exact & $1+\mathrm{CV}^2$ & $1/\varphi$ & exact & $1+\mathrm{CV}^2$ & exact \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}""")


def tab_robust(d):
    rows = []
    for sy in d["systems"]:
        g = sy["grid"]
        ab = [r["rules"]["abs"]["ratio"] for r in g if r["rules"]["abs"].get("reachable", True)]
        rl = [r["rules"]["rel"]["ratio"] for r in g]
        lm = [r["rules"]["lam"]["ratio"] for r in g]
        inf = max(100 * r["rules"]["abs"]["infeasible_mass"] for r in g)
        nz = sum(1 for r in g if r["rules"]["abs"]["infeasible_mass"] == 0)
        name = sy["name"].replace("&", "\\&")
        rows.append(f"{name} & {sy['p']:.3f} & {min(ab):.2f}--{max(ab):.2f} & {min(rl):.2f}--{max(rl):.2f} & "
                    f"{min(lm):.3f}--{max(lm):.3f} & {inf:.0f}\\% & {nz}/{len(g)} \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Eight small systems. For each, the range over the target grid of the KL multiple (against the minimum) of the fixed budget (FB), the always-feasible budget (AF) and the fixed price (FP); the largest share of node visits at which the FB budget is infeasible; and the number of targets with no infeasible node. Targets are $p\times\{0.85,0.7,0.5,0.3,0.15,0.05,0.02\}$ (the base system uses the grid of Table~\ref{tab:ladder} and three near-reference targets). ``Persistent'': $R$ consecutive repeats are required for a loop.}
\label{tab:robust}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
System & $p$ & FB & AF & FP & max infeasible & targets with 0\% infeasible \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}""")


def _commonkl_rows():
    return {r["strategy"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "runs/common_kl_comparison_v4_samples/common_kl_comparison.csv")))}


def tab_primary():
    rows = _commonkl_rows()
    order = [("temperature_0.10", "Temperature 0.10 (reference)"), ("rr_fixed", r"\textsc{rr-fixed} ($\lambda{=}2$)"),
             ("rr_adaptive", r"\textsc{rr-adaptive}"), ("lzpenalty", "LZ Penalty (adapted)"),
             ("rep_penalty_1.3", "Repetition penalty 1.3"), ("suffixmatch_a3.0", r"SuffixMatch ($\alpha{=}3$)"),
             ("fsd", "FSD-style (adapted)"), ("no_repeat_4gram", "No-repeat 4-gram$^{*}$"),
             ("rr_dual_eps0.05", r"\textsc{rr-dual} ($\varepsilon{=}0.05$)"),
             ("rr_dual_eps0.01", r"\textsc{rr-dual} ($\varepsilon{=}0.01$)")]
    lines = []
    for key, label in order:
        r = rows[key]
        n = int(r["n_samples"])
        k = round(float(r["loop_rate"]) * n)
        lines.append(f"{label} & {k}/{n} & {float(r['kl_bits_mean']):.3f} & {float(r['risk_achieved_mean']):.3f} \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Primary model (\textsc{cosine}), $T{=}0.10$, 15 test prompts $\times$ 10 seeds ($n{=}150$ each). Loops: generations in which the detector fires. $\KL(q\Vert p)$: per-step mean in bits against the common reference $p$ (multiply by 500 for the path KL of one generation); $\E_q[\mathrm{risk}]$: achieved recurrence risk under the same risk definition for every row. Sorted by KL. RMST, mcNLL and 4-gram similarity are in Table~\ref{tab:main}.}
\label{tab:primary}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Loops & $\KL(q\Vert p)$ (bits) & $\E_q[\mathrm{risk}]$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
{\footnotesize $^{*}$Hard constraint on the measured event, so its zero loops are not a finding (Appendix~\ref{ssec:hard-repeat}). 95\% cluster CIs for KL and risk are within $\pm$6\% of the mean shown.}
\end{table}""")


def tab_pcb():
    d = json.load(open(os.path.join(ROOT, "runs/common_kl_comparison_v4_samples/prompt_conditional_bound.json")))
    d.pop("_reference")
    names = [("rr_fixed", r"\textsc{rr-fixed}"), ("rr_adaptive", r"\textsc{rr-adaptive}"),
             ("lzpenalty", "LZ Penalty (adapted)"), ("rep_penalty_1.3", "Repetition penalty 1.3"),
             ("suffixmatch_a3.0", r"SuffixMatch ($\alpha{=}3$)"), ("fsd", "FSD-style (adapted)"),
             ("no_repeat_4gram", "No-repeat 4-gram"),
             ("rr_dual_eps0.05", r"\textsc{rr-dual} ($\varepsilon{=}0.05$)"),
             ("rr_dual_eps0.01", r"\textsc{rr-dual} ($\varepsilon{=}0.01$)")]
    rows = []
    for key, label in names:
        r = d[key]
        se = r["sensitivity"]
        vals = [se["laplace"]["B_prompt_conditional"], r["bound_prompt_conditional"], se["plug_in"]["B_prompt_conditional"]]
        rows.append(f"{label} & {r['loop_events']}/{r['n_samples']} & {r['path_kl_bits']:.1f} [{r['path_kl_ci95'][0]:.1f}, {r['path_kl_ci95'][1]:.1f}] "
                    f"& {r['bound_prompt_conditional']:.2f} [{r['bound_ci95'][0]:.2f}, {r['bound_ci95'][1]:.2f}] & {min(vals):.2f}--{max(vals):.2f} "
                    f"& {r['excess_bits']:.1f} [{r['excess_ci95'][0]:.1f}, {r['excess_ci95'][1]:.1f}] \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Prompt-conditional decomposition of total path KL (bits per generation of 500 characters) into the event term $\frac1{15}\sum_x d_{\mathrm{Ber}}(\tilde Q_x(E)\Vert\tilde P_x(E))$ and the remainder (excess), primary model, $T{=}0.10$, 150 generations per decoder. Brackets are 95\% percentile intervals from resampling the 15 prompts. A tilde marks Jeffreys-smoothed loop fractions over 10 seeds; ``range'' gives the event term under Laplace, Jeffreys and plug-in fractions. Descriptive plug-in estimates, not bounds with guaranteed coverage.}
\label{tab:pcb}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Method & Loops & Path KL & Event term & Range & Excess \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}""")


def frontier_rows():
    paths = sorted(glob.glob(os.path.join(ROOT, "runs/frontier_v1*/frontier_rows.csv")))
    rows = []
    for pth in paths:
        rows += list(csv.DictReader(open(pth)))
    return rows


def tab_frontier(rows):
    def key(r):
        m = re.search(r"(lam|eps)([0-9.]+)", r["strategy"])
        return (0 if r["strategy"].startswith("rr_fixed") else 1, float(m.group(2)) * (1 if r["strategy"].startswith("rr_fixed") else -1) if m else 0)
    ref = [r for r in rows if r["strategy"] == "temperature_0.10"]
    fam = sorted([r for r in rows if r["strategy"] != "temperature_0.10"], key=key)
    lines = []
    for r in ref + fam:
        s = r["strategy"]
        if s == "temperature_0.10":
            lab, lm = "reference (T=0.10)", "---"
        elif s.startswith("rr_fixed"):
            lab = r"fixed price, $\lambda{=}%s$" % re.search(r"lam([0-9.]+)", s).group(1)
            lm = "%s" % ("%.1f" % float(r["lam_mean"]))
        else:
            lab = r"fixed budget, $\varepsilon{=}%s$" % re.search(r"eps([0-9.]+)", s).group(1)
            lm = "%.1f / %.1f" % (float(r["lam_mean"]), float(r["lam_std"]))
        n = int(r["n_samples"])
        lines.append(f"{lab} & {int(r['loop_count'])}/{n} & {float(r['kl_bits_total_mean']):.1f} "
                     f"& {lm} & {100*float(r['lam_frac_zero']) if r['strategy']!='temperature_0.10' else 0:.0f}\\% \\\\")
    return (r"""\begin{table}[H]
\centering
\caption{Primary model: fixed-price (recurrence-risk \textsc{fixed}) and fixed-budget (\textsc{dual}) rules at matched settings of the same sweep. Loops: generations with a detected loop. KL: path KL per 500-character generation, bits. Multiplier: mean (and, for the budget rule, mean / s.d.) of the per-step multiplier $\lambda_t$ realized over the generation, in nats of risk. $\lambda_t=0$: share of steps at which the budget constraint was inactive.}
\label{tab:frontier}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Rule & Loops & Path KL (bits) & Multiplier & $\lambda_t=0$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}""")


def tab_transfer(frontier_status):
    s2, s3 = frontier_status
    return (r"""\begin{table}[H]
\centering
\caption{Findings of the exact systems against the language-model evidence.}
\label{tab:transfer}
\small
\begin{tabular}{@{}p{0.27\linewidth}p{0.45\linewidth}p{0.20\linewidth}@{}}
\toprule
Finding in the exact systems & Language-model evidence & Status \\
\midrule
A fixed budget spends far more KL than the minimum for the loop probability it reaches & Every decoder, \textsc{rr-dual} included, spends at least 98\% of its path KL beyond the event term (Table~\ref{tab:pcb}) & Consistent; no saving is shown \\
At equal loop rate a fixed price is cheaper than a fixed budget & Sweep of Section~\ref{ssec:frontier} (Figure~\ref{fig:frontier}, Table~\ref{tab:frontier}) & """ + s2 + r""" \\
A fixed budget realizes widely varying prices & Per-step multiplier traces of \textsc{rr-dual} (Table~\ref{tab:frontier}) & """ + s3 + r""" \\
The waste does not need infeasible steps & \textsc{rr-dual} is feasible at 100\% of steps on every model (Appendix~\ref{sec:generalization}) & Supported for the recurrence-risk proxy \\
With the exact hazard a causal rule attains the optimum & A rollout estimate separates risky from novel candidates; no decoder is built on it (Appendix~\ref{sec:hazard-transfer}) & Hypothesis \\
The quantitative waste law & Needs the hazard as risk signal, which is not available & Hypothesis \\
Rejection of looping samples attains the bound when streaming is not needed & Cost of 0.49 bits and 1.4 generations computed from the loop rate; not run end to end & Computed, not tested \\
\bottomrule
\end{tabular}
\end{table}""")


# ----------------------------------------------------------------------------- bibliography
def bib_to_mdpi(path):
    txt = open(path).read()
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}\n", txt, re.S):
        typ, key, body = m.groups()
        f = {x.group(1).lower(): x.group(2) for x in re.finditer(r"(\w+)\s*=\s*\{(.*)\},?\s*$", body, re.M)}
        entries.append((typ, key, f))

    def names(a):
        out = []
        for person in a.split(" and "):
            last, first = [x.strip() for x in person.split(",")] if "," in person else (person, "")
            ini = "".join(w[0] + "." for w in re.split(r"[ \-]", first) if w and w[0].isalpha())
            out.append(f"{last}, {ini}")
        return "; ".join(out)

    out = []
    for typ, key, f in sorted(entries, key=lambda e: e[2]["author"].lower()):
        au = names(f["author"])
        title = f["title"]
        y = f["year"]
        if typ == "book":
            ed = f", {f['edition']} ed." if "edition" in f else ""
            line = f"{au} \\textit{{{title}}}{ed}; {f['publisher']}, {y}."
        elif typ == "inproceedings":
            extra = f", pp. {f['pages']}" if "pages" in f else ""
            ser = f", PMLR {f['volume']}" if f.get("series") else ""
            line = f"{au} {title}. In \\textit{{{f['booktitle']}}}{ser}{extra}; {y}."
            if "note" in f:
                line += f" {f['note']}."
        else:
            vol = ""
            if "volume" in f:
                vol = f" \\textbf{{{y}}}, \\textit{{{f['volume']}}}"
                if "number" in f:
                    vol += f"({f['number']})"
                if "pages" in f:
                    vol += f", {f['pages']}"
                vol += "."
            else:
                vol = f" \\textbf{{{y}}}."
            line = f"{au} {title}. \\textit{{{f['journal']}}}{vol}"
            if "note" in f:
                line += f" {f['note']}."
        out.append(f"\\bibitem{{{key}}}\n{line}\n")
    return "\\begin{thebibliography}{999}\n" + "\n".join(out) + "\\end{thebibliography}"


# ----------------------------------------------------------------------------- assembly
LABEL_MAP = [("cor:affine", "prop:causal"), ("sec:results", "ssec:common-kl"), ("ssec:gap", "ssec:ladder"),
             ("tab:ablation", "tab:ladder"), ("tab:commonkl", "tab:primary"),
             ("sec:formal", "sec:setup"), ("sec:synthetic", "sec:small")]


def fix_refs(s):
    for a, b in LABEL_MAP:
        s = s.replace("{" + a + "}", "{" + b + "}")
    return s


FRONT = r"""\documentclass[make,article,submit,pdftex]{Definitions/mdpi}

\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{float}
\usepackage{enumitem}
\providecommand{\KL}{}\renewcommand{\KL}{\mathrm{KL}}
\providecommand{\E}{}\renewcommand{\E}{\mathbb{E}}
\newcommand{\risk}{\mathrm{risk}}
\newcommand{\haz}{\mathrm{haz}}
\newtheorem{prop}{Proposition}
\setlength{\emergencystretch}{3em}
% macros that some versions of the MDPI class do not define (no-ops where undefined)
\providecommand{\hreflink}[1]{}
\providecommand{\TitleCitation}[1]{}
\providecommand{\AuthorCitation}[1]{}

%% Front matter -- fill in the author block before submission.
\firstpage{1}
\makeatletter
\setcounter{page}{\@firstpage}
\makeatother
\pubvolume{1}
\issuenum{1}
\articlenumber{0}
\pubyear{2026}
\copyrightyear{2026}
\datereceived{}
\daterevised{}
\dateaccepted{}
\datepublished{}
\hreflink{}

\Title{Why Fixed Per-Step Risk Budgets Waste KL When Preventing Repetition Loops}
\TitleCitation{Why Fixed Per-Step Risk Budgets Waste KL When Preventing Repetition Loops}
\Author{[Author Name] $^{1}$}
\AuthorNames{[Author Name]}
\AuthorCitation{[Author Surname], [I.]}
\address{%
$^{1}$ \quad [Affiliation, city, country]}
\corres{Correspondence: [email address]}

\abstract{{{ABSTRACT}}}

\keyword{repetition loops; decoding; KL divergence; Lagrange multiplier; constrained decoding; language models}

\begin{document}
"""

BACK = r"""
\vspace{6pt}
\authorcontributions{The author conceived the study, derived the results, wrote the code, ran the experiments and wrote the manuscript.}
\funding{This research received no external funding.}
\institutionalreview{No institutional ethical review was sought. The pilot human evaluation of Appendix~\\ref{sec:human-eval} was a short anonymous task in which volunteers judged machine-generated text; no sensitive personal data were collected, and the released result files identify participants only by the labels they chose.}
\informedconsent{Participants were told the nature and length of the task on the introduction page of the evaluation tool before taking part; no written consent was collected.}
\dataavailability{Code, tests, saved results and the scripts that generate every table are available at https://github.com/inopolis/GAI-project (release tag \\texttt{paper-final}); see Appendix~\\ref{sec:reproducibility}.}
\conflictsofinterest{The author declares no conflict of interest.}

\appendixtitles{yes}
\appendixstart
\appendix
"""


ARTICLE_FRONT = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,booktabs,multirow,graphicx,float,enumitem,hyperref,xcolor,caption}
\providecommand{\KL}{}\renewcommand{\KL}{\mathrm{KL}}
\providecommand{\E}{}\renewcommand{\E}{\mathbb{E}}
\newcommand{\risk}{\mathrm{risk}}
\newcommand{\haz}{\mathrm{haz}}
\newtheorem{prop}{Proposition}
\setlength{\emergencystretch}{3em}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
% stand-ins for the MDPI macros so that the same body compiles with a plain article class
\newenvironment{adjustwidth}[2]{}{}
\newcommand{\extralength}{0pt}
\newcommand{\reftitle}[1]{\section*{#1}}
\newcommand{\appendixtitles}[1]{}
\newcommand{\appendixstart}{}
\newcommand{\authorcontributions}[1]{\paragraph{Author contributions.} #1}
\newcommand{\funding}[1]{\paragraph{Funding.} #1}
\newcommand{\institutionalreview}[1]{\paragraph{Institutional review.} #1}
\newcommand{\informedconsent}[1]{\paragraph{Informed consent.} #1}
\newcommand{\dataavailability}[1]{\paragraph{Data availability.} #1}
\newcommand{\conflictsofinterest}[1]{\paragraph{Conflicts of interest.} #1}
\title{Why Fixed Per-Step Risk Budgets Waste KL When Preventing Repetition Loops}
\author{}
\date{}
\begin{document}
\maketitle
\begin{abstract}
{{ABSTRACT}}
\end{abstract}
\tableofcontents
"""


def make_manifest_appendix():
    old = rd("appendix_manifest_old.tex") if os.path.exists(os.path.join(SRC, "appendix_manifest_old.tex")) else ""
    return old


def build(out, article=False):
    d = json.load(open(os.path.join(ROOT, "budget_ladder.json")))
    frows = frontier_rows()
    pending = "\\textbf{[pending: frontier run]}"
    tables = {"ladder": tab_ladder(d), "smalld": tab_smalld(d), "robust": tab_robust(d),
              "primary": tab_primary(), "pcb": tab_pcb()}
    if frows:
        tables["frontier"] = tab_frontier(frows)
        status = (rd("frontier_status.txt").strip().split("\n") + ["", ""])[:2]
    else:
        status = [pending, pending]
    tables["transfer"] = tab_transfer(status)

    abstract = rd("abstract.tex").strip() if os.path.exists(os.path.join(SRC, "abstract.tex")) else pending
    front = (ARTICLE_FRONT if article else FRONT).replace("{{ABSTRACT}}", abstract)

    main = ""
    for f in ["01_intro.tex", "02_related.tex", "03_setup.tex", "04_theory.tex", "05_small.tex",
              "06_real.tex", "07_discussion.tex"]:
        main += rd(f) + "\n"
    frontier_text = rd("frontier_results.tex") if os.path.exists(os.path.join(SRC, "frontier_results.tex")) else pending
    main = main.replace("{{FRONTIER_RESULTS}}", frontier_text)
    concl = rd("conclusion_lm.txt").strip() if os.path.exists(os.path.join(SRC, "conclusion_lm.txt")) else pending
    main = main.replace("{{CONCLUSION_LM}}", concl)
    for name, t in tables.items():
        main = main.replace("{{TABLE:%s}}" % name, t)
    main = fix_refs(main)

    app = ""
    for f in ["appendix_proofs.tex", "appendix_event.tex", "appendix_protocol.tex", "appendix_baselines.tex",
              "appendix_models.tex", "appendix_indicator.tex", "appendix_human.tex", "appendix_hazard.tex",
              "appendix_defects.tex", "appendix_manifest.tex", "appendix_prompts.tex"]:
        app += fix_refs(rd(f)) + "\n"
    app = app.replace("{{TABLE:main}}", rd("old_table_main.tex"))
    app = app.replace("{{TABLE:oracle}}", rd("old_table_oracle.tex")).replace("{{TABLE:synthetic}}", rd("old_table_synthetic.tex"))
    doc = front + main + BACK + app + "\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\reftitle{References}\n" \
          + bib_to_mdpi(os.path.join(ROOT, "references.bib")) + "\n\\end{adjustwidth}\n\\end{document}\n"
    doc = re.sub(r"\\begin\{table\}\[(?:ht|h|ht!|H)\]", r"\\begin{table}[H]", doc)
    doc = re.sub(r"\\cite[pt]\*?(?:\[[^\]]*\])*\{", r"\\cite{", doc)   # natbib commands of older text -> numeric \cite
    open(out, "w").write(doc)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper_make.tex"))
    ap.add_argument("--article", action="store_true", help="plain article class (for a local compile check)")
    a = ap.parse_args()
    doc = build(a.out, article=a.article)
    print("wrote", a.out, len(doc.split()), "words")
