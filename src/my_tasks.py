"""Build MY_TASKS.md: the one list of everything the USER has to do themselves.

Called by update_results.py every time it runs (you do not run this file directly).

What it reads
-------------
  - the decisions parsed from DECISIONS.md      -> "Decisions to review"
  - the run records in results/ and configs/    -> "Runs to do"
  - the "My notes" blocks of RESULTS.md         -> "Results to interpret"
  - the Checks results                          -> "Failed checks"
  - git (uncommitted files, unpushed commits)   -> "Waiting on me now"
  - the settings in configs/update_results.json -> deadline, schedule, requirement list

Rules it follows
----------------
  * Every item is a checkbox `- [ ]`. This script NEVER writes `[x]`. When the file
    is rebuilt, a box you ticked stays ticked (matched by the hidden `<!--id:...-->`
    tag at the end of the line, so do not delete those tags).
  * Under each section there is an "Added by me" block between two marker comments.
    Write anything there; it is copied back unchanged on every rebuild.
  * An item is overdue when its planned day (from `schedule` in the config) is before
    today in Singapore time and its box is not ticked. Overdue items are listed first.
  * Items whose condition no longer holds (a decision got resolved, a run now has
    results, ...) disappear on the next rebuild, whether ticked or not.
"""

import glob
import json
import os
import re
import subprocess
from datetime import date, datetime
from zoneinfo import ZoneInfo

from runconfig import REPO_ROOT

SECTIONS = [("waiting", "Waiting on me now"), ("decisions", "Decisions to review"), ("runs", "Runs to do"),
            ("interpret", "Results to interpret"), ("report", "Report and submission"), ("failed", "Failed checks")]
PART_TITLES = {"data": ("Data", "data"), "part1": ("Part I", "part-i"), "part2": ("Part II", "part-ii"),
               "part3": ("Part III", "part-iii")}


# ----------------------------------------------------------------------- git
def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state():
    """(number of uncommitted files, number of unpushed commits or None, remote url or None)."""
    status = _git("status", "--porcelain")
    uncommitted = len(status.splitlines()) if status else 0
    ahead = _git("rev-list", "--count", "@{u}..HEAD")
    return uncommitted, (int(ahead) if ahead is not None else None), _git("remote", "get-url", "origin")


# ----------------------------------------------------------------- old file
def read_ticks_and_additions(path, force):
    """({item id: True} for boxes ticked in the old file, {section key: your text})."""
    if not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as f:
        old = f.read()
    ticks = {}
    for line in old.splitlines():
        m = re.match(r"^- \[[xX]\] .*<!--id:(\S+?)-->\s*$", line)
        if m:
            ticks[m.group(1)] = True
    additions = {}
    for key, _ in SECTIONS:
        m = re.search(rf"<!-- my-tasks:{key}:start -->\n(.*?)<!-- my-tasks:{key}:end -->", old, re.S)
        if not m:
            if force:
                return ticks, {}
            raise SystemExit(f"{path} no longer has all the my-tasks marker lines, so what you wrote might be lost. "
                             "Restore the markers (or run with --force to discard it).")
        additions[key] = m.group(1)
    return ticks, additions


# --------------------------------------------------------------------- items
def make_item(item_id, text, details=None, section_key=None):
    return {"id": item_id, "text": text, "details": details or [], "section": section_key}


def planned_day(item, schedule):
    """date or None: the item's own entry in the schedule, else its section's default."""
    for key in (item["id"], f"section:{item['section']}"):
        if schedule.get(key):
            return date.fromisoformat(schedule[key])
    return None


def pending_runs(opts, runs):
    """Configs in configs/ that have no results/<run>.json yet (and are not on the ignore list)."""
    pending = []
    for path in sorted(glob.glob(os.path.join(opts["configs_dir"], "*.json"))):
        run = os.path.splitext(os.path.basename(path))[0]
        if run in opts["tasks_ignore_configs"] or run in runs:
            continue
        pending.append(run)
    return pending


def colab_only_local_runs(opts, runs):
    """Runs of a Colab-only script (neural training, benchmark, feature extraction, neural generation)
    whose record shows they ran on a CPU, and that have no Colab re-run record yet.

    Returns (run, "script --config configs/<run>.json [--model ...]", record) tuples.
    """
    found = []
    for run, rec in runs.items():
        script = rec.get("script", "")
        neural_generation = script == opts["colab_only_generate_script"] and \
            (rec["config"].get("options") or {}).get("model") != opts["colab_only_generate_skip_model"]
        if run.startswith(opts["test_run_prefix"]) or run.endswith(opts["colab_rerun_suffix"]):
            continue
        if script not in opts["colab_only_scripts"] and not neural_generation:
            continue
        if not str(rec.get("hardware", "")).startswith("CPU") or f"{run}{opts['colab_rerun_suffix']}" in runs:
            continue
        model_flag = f" --model {rec['config']['architecture']}" if script == "src/train.py" else ""
        found.append((run, f"{script} --config configs/{run}.json{model_flag}", rec))
    return found


def script_for(run, rules):
    for candidate in (f"src/{run}.py", f"tests/{run}.py"):
        if os.path.exists(os.path.join(REPO_ROOT, candidate)):
            return candidate
    for pattern, script in rules:
        if re.search(pattern, run):
            return script
    return None


def time_estimate(script, runs):
    """Fact from earlier runs of the same script, or a statement that none is recorded."""
    for run, rec in runs.items():
        if rec.get("script") != script:
            continue
        seconds = (rec.get("metrics") or {}).get("train_seconds") or (rec.get("details") or {}).get("train_seconds")
        if seconds:
            return f"earlier run `{run}` of this script took {seconds:,.0f} s (about {seconds / 3600:.1f} h) on {rec.get('hardware')}"
    return "no earlier run of this script has a recorded time"


def requirement_status(req, runs, ticked, git, checks):
    """(is_done, evidence text) for one report/submission requirement."""
    kind = req["evidence"]
    if kind == "file_regex":
        path = os.path.join(REPO_ROOT, req["file"])
        text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        missing = [r for r in req["regexes"] if not re.search(r, text)]
        return (not missing), (f"`{req['file']}` matches every pattern {req['regexes']} (keywords only; completeness is not checked)"
                               if not missing else f"`{req['file']}` has no match for {missing}")
    if kind == "architectures":
        have = {rec["config"].get("architecture") or "" for r, rec in runs.items() if not r.startswith("test_")}
        missing = [name for name, pattern in req["required"].items() if not any(re.search(pattern, a, re.I) for a in have)]
        return (not missing), (f"all present: {', '.join(req['required'])}" if not missing else
                               f"no run yet for: {', '.join(missing)} (architectures with results: {', '.join(sorted(a for a in have if a))})")
    if kind == "models":
        names = [r for r, rec in runs.items() if not r.startswith("test_") and not rec.get("generations")
                 and ("test_perplexity" in (rec.get("metrics") or {}) or "test" in (rec.get("metrics") or {}))]
        return False, f"{len(names)} model(s) with a test result so far: {', '.join(f'`{n}`' for n in names) or 'none'}"
    if kind == "check":
        status = next((res[0] for name, res in checks if name == req["name"]), "unknown")
        return status == "PASS", f"Checks section: \"{req['name']}\" is {status}"
    if kind == "repo_access":
        uncommitted, ahead, remote = git
        return False, (f"remote `origin` = {remote or 'none'}; {uncommitted} uncommitted file(s); "
                       f"{'unknown' if ahead is None else ahead} unpushed commit(s). Whether the instructor has access cannot be checked from here")
    return (req["id"] in ticked), ("ticked by you" if req["id"] in ticked else "cannot be checked automatically; tick when finished")


# --------------------------------------------------------------------- build
def build_items(opts, runs, decisions, checks, notes, section_runs, git):
    items = {key: [] for key, _ in SECTIONS}
    needs = [d for d in decisions if "NEEDS REVIEW" in d["Status"]]
    uncommitted, ahead, _ = git
    pending = pending_runs(opts, runs)
    failed = [(name, res) for name, res in checks if res[0] == "FAIL"]
    pending_notes = [k for k in PART_TITLES if section_runs.get(k) and not (notes.get(k) or "").strip()]

    # --- Waiting on me now: things that block the next step ------------------
    w = items["waiting"]
    if not opts["schedule"]:
        w.append(make_item("wait-schedule", "Put your planned day for each task into `schedule` in `configs/update_results.json` "
                           "(item id -> `YYYY-MM-DD`, or `section:<key>` -> date). Until then every item shows \"Planned: not set\" and nothing can be overdue",
                           section_key="waiting"))
    if needs:
        w.append(make_item("wait-decisions", f"Review {len(needs)} NEEDS REVIEW decision(s): see [Decisions to review](#decisions-to-review)",
                           section_key="waiting"))
    for run, _, _ in colab_only_local_runs(opts, runs):
        w.append(make_item(f"wait-rerun-{run}", f"Re-run `{run}` in Colab (it ran locally): see [Runs to do](#runs-to-do)", section_key="waiting"))
    for run in pending:
        w.append(make_item(f"wait-run-{run}", f"Start run `{run}`: see [Runs to do](#runs-to-do)", section_key="waiting"))
    if pending_notes:
        w.append(make_item("wait-notes", f"Write your notes for {', '.join(PART_TITLES[k][0] for k in pending_notes)}: "
                           "see [Results to interpret](#results-to-interpret)", section_key="waiting"))
    for name, res in failed:
        w.append(make_item(f"wait-fail-{re.sub('[^a-z0-9]+', '-', name.lower()).strip('-')}", f"Fix failing check: {name}: see [Failed checks](#failed-checks)",
                           section_key="waiting"))
    if uncommitted or ahead:
        w.append(make_item("wait-commit", f"Commit and push: {uncommitted} uncommitted file(s), "
                           f"{'unknown number of' if ahead is None else ahead} unpushed commit(s). The instructor can only see what is pushed",
                           section_key="waiting"))

    # --- Decisions to review --------------------------------------------------
    for d in needs:
        options = d["Alternatives"] if d["Alternatives"] != "not stated" else "options not recorded in DECISIONS.md"
        options = options if len(options) <= 220 else options[:217].rsplit(" ", 1)[0] + "..."
        items["decisions"].append(make_item(
            f"dec-{d['id']}", f"**{d['id']}** {d['title']}",
            [f"Options: {options}", f"Current value: {d['Value'][:160]}{'...' if len(d['Value']) > 160 else ''}",
             f"Record your choice: tell Claude in chat; it is written to `{opts['decisions_path']}` entry {d['id']} (line {d['line']}), "
             "which sets `Chosen by` and `Status`"], "decisions"))

    # --- Runs to do -----------------------------------------------------------
    for run in pending:
        script = script_for(run, opts["run_script_rules"])
        command = f"python3 {script} --config configs/{run}.json" if script else f"(no script known for configs/{run}.json)"
        items["runs"].append(make_item(
            f"run-{run}", f"Run `{run}`",
            [f"Command: `{command}` (in Colab: the same command with a leading `!`, from inside the repo folder)",
             f"Time: {time_estimate(script, runs)}",
             f"Produces: `results/{run}.json`, then a new section in `RESULTS.md` after `update_results.py`"], "runs"))
    for extra in opts["extra_items"]:  # items from the assignment brief that scripts cannot detect
        # `done_if_result`: drop the item once results/<that run>.json exists.
        if extra.get("done_if_result") and extra["done_if_result"] in runs:
            continue
        items[extra["section"]].append(make_item(extra["id"], extra["text"], extra.get("details", []), extra["section"]))
    for run, command_line, rec in colab_only_local_runs(opts, runs):
        suffix = opts["colab_rerun_suffix"]
        seconds = (rec.get("metrics") or {}).get("train_seconds")
        items["runs"].append(make_item(
            f"rerun-{run}", f"Re-run `{run}` in Colab: it was run locally on {rec.get('hardware')}, and Colab-only runs must come from the Colab GPU",
            [f"Command (Colab cell): `!python -u {command_line} --run-name {run}{suffix}` (a new run name, so the local result is kept)",
             f"Time: the local run took {seconds:,.0f} s (about {seconds / 3600:.1f} h) on CPU; GPU time not known yet" if seconds
             else "Time: not known yet",
             f"Produces: `results/{run}{suffix}.json`, `checkpoints/{run}{suffix}.pt`; the local `results/{run}.json` stays as it is"], "runs"))

    # --- Results to interpret ------------------------------------------------
    for key in pending_notes:
        title, anchor = PART_TITLES[key]
        items["interpret"].append(make_item(
            f"interp-{key}", f"Write your notes for [{title}](RESULTS.md#{anchor})",
            [f"Runs in this section: {', '.join(f'`{r}`' for r in section_runs[key])}",
             f"Where: the empty *My notes* block at the end of the {title} section of `RESULTS.md`"], "interpret"))

    # --- Report and submission ------------------------------------------------
    return items, git, failed


def render(opts, items, ticks, additions, now, runs, git, failed):
    deadline = datetime.fromisoformat(opts["deadline"])
    today = now.date()
    days_left = (deadline.date() - today).days
    schedule = opts["schedule"]
    out = ["# My tasks\n",
           f"Rebuilt by `src/update_results.py` on {now.strftime('%a %d %b %Y, %H:%M')} Singapore time. "
           f"**Deadline: {deadline.strftime('%a %d %b %Y, %I:%M %p').replace(' 0', ' ')} SGT** "
           f"({days_left} day(s) left; today is {now.strftime('%a %d %b')}).\n",
           "Tick a box yourself with `[x]`; this file never ticks anything for you, and keeps your ticks and anything you write "
           "between the marker lines. Do not delete the `<!--id:...-->` tags at the end of each line.\n",
           "Contents: " + " · ".join(f"[{t}](#{t.lower().replace(' ', '-')})" for _, t in SECTIONS) + "\n"]
    for key, title in SECTIONS:
        out.append(f"## {title}\n")
        entries = []
        for position, item in enumerate(items[key]):
            planned = planned_day(item, schedule)
            ticked = item["id"] in ticks
            overdue = planned is not None and planned < today and not ticked
            entries.append((not overdue, planned or date.max, position, item, planned, ticked, overdue))
        # Overdue first (earliest planned day first), then everything else in its natural order.
        entries.sort(key=lambda e: (e[0], e[1] if not e[0] else date.max, e[2]))
        if not entries:
            out.append("Nothing here right now.\n")
        for _, _, _, item, planned, ticked, overdue in entries:
            plan = f"Planned: {planned.strftime('%a %d %b')}" if planned else "Planned: not set"
            flag = "**OVERDUE** " if overdue else ""
            out.append(f"- [{'x' if ticked else ' '}] {flag}{item['text']} *({plan})* <!--id:{item['id']}-->")
            for detail in item["details"]:
                out.append(f"  - {detail}")
        out.append("")
        out.append("### Added by me\n")
        out.append(f"<!-- my-tasks:{key}:start -->\n{additions.get(key, chr(10))}<!-- my-tasks:{key}:end -->\n")
    return "\n".join(out)


def write_my_tasks(opts, runs, decisions, checks, notes, section_runs, force=False):
    now = datetime.now(ZoneInfo(opts["timezone"]))
    git = git_state()
    ticks, additions = read_ticks_and_additions(opts["tasks_output_path"], force)
    items, git, failed = build_items(opts, runs, decisions, checks, notes or {}, section_runs, git)

    # Report and submission: needs the ticks (for the manual items), so it is built here.
    for req in opts["report_requirements"]:
        done, evidence = requirement_status(req, runs, ticks, git, checks)
        hint = f"Where to look: {req['hint']}" if req.get("hint") else None
        items["report"].append(make_item(
            req["id"], req["text"],
            [f"Status: {'done' if done else 'not done'} ({evidence})"] + ([hint] if hint else []), "report"))

    # Failed checks section.
    for name, (status, evidence) in failed:
        items["failed"].append(make_item(f"fail-{re.sub('[^a-z0-9]+', '-', name.lower()).strip('-')}",
                                         f"{name}: FAIL", [f"Evidence (Checks section of `RESULTS.md`): {e}" for e in evidence]
                                         + ["Needs fixing: the evidence above shows what does not hold; after fixing, re-run `update_results.py`"],
                                         "failed"))

    text = render(opts, items, ticks, additions, now, runs, git, failed)
    with open(opts["tasks_output_path"], "w", encoding="utf-8") as f:
        f.write(text)
    return opts["tasks_output_path"]
