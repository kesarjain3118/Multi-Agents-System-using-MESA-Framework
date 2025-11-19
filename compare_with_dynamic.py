# compare_with_dynamic.py  (final — overwrite your existing file with this)
import os
import random
import importlib
import traceback
import time
import sys
import contextlib

import pandas as pd
import matplotlib.pyplot as plt

from comparative_pathfinding import run_experiment

# ========== PARAMETERS ==========
RUNS_BASELINE = 2        # reduce for quicker tests
RUNS_DYNAMIC = 2
WIDTH = 20
HEIGHT = 20
N_AGENTS = 10
N_OBSTACLES = 100
MAX_STEPS = 400          # hard cap for each dynamic run
STAGNATION_WINDOW = 100  # if no agent moves for this many consecutive model.step() calls -> break
OUT_CSV = os.path.join(os.getcwd(), "combined_with_dynamic.csv")
OUT_CSV_CLEAN = os.path.join(os.getcwd(), "combined_with_dynamic_cleaned.csv")
PLOT_PREFIX = os.path.join(os.getcwd(), "compare_plot_")

print("Working dir:", os.getcwd())
print("Will write combined CSV to:", OUT_CSV)
print("Params:", dict(RUNS_BASELINE=RUNS_BASELINE, RUNS_DYNAMIC=RUNS_DYNAMIC,
                     WIDTH=WIDTH, HEIGHT=HEIGHT, N_AGENTS=N_AGENTS, N_OBSTACLES=N_OBSTACLES,
                     MAX_STEPS=MAX_STEPS, STAGNATION_WINDOW=STAGNATION_WINDOW))
print()

# --------- 1) run baseline experiment(s) ----------
modes = ["greedy", "astar", "pso_only", "rl_only", "hybrid"]
all_dfs = []
for mode in modes:
    print(f"--- baseline: {mode} ---")
    df, summary = run_experiment(mode=mode, n_runs=RUNS_BASELINE,
                                width=WIDTH, height=HEIGHT, n_agents=N_AGENTS,
                                n_obstacles=N_OBSTACLES, max_steps=MAX_STEPS)
    # run_experiment returns dataframe with a 'mode' column — keep that, but standardize below
    df["model"] = mode
    all_dfs.append(df)
    print(summary)
    print()

# --------- 2) import dynamic model ----------
DYN_MODULE = "Dynamic_hybrid_model"
DYN_CLASS = "PathFindingModel"
try:
    mod = importlib.import_module(DYN_MODULE)
    ModelClass = getattr(mod, DYN_CLASS)
    print("Imported dynamic model:", DYN_MODULE + "." + DYN_CLASS)
except Exception as e:
    print("Failed to import dynamic model:", e)
    traceback.print_exc()
    raise

# --------- helper: prevent plotting from blocking & close figures ----------
_real_show = plt.show
def _noop_show(*args, **kwargs):
    return None
plt.show = _noop_show  # disable blocking show

# --------- 3) run dynamic model trials with stagnation detection ----------
dyn_rows = []
for run_idx in range(RUNS_DYNAMIC):
    seed = random.randint(0, 10**6)
    print(f"\nDynamic run {run_idx+1}/{RUNS_DYNAMIC} (seed={seed})")
    # instantiate model (flexible constructor)
    try:
        model = ModelClass(width=WIDTH, height=HEIGHT, n_agents=N_AGENTS, n_obstacles=N_OBSTACLES)
    except TypeError:
        model = ModelClass(WIDTH, HEIGHT, N_AGENTS, N_OBSTACLES)

    # ensure model writes accuracy to a safe local path
    model.accuracy_file = os.path.join(os.getcwd(), f"simulation_accuracy_log_dynamic_run{run_idx+1}.csv")
    # ensure file has headers to avoid pandas EmptyDataError
    if not os.path.exists(model.accuracy_file) or os.path.getsize(model.accuracy_file) == 0:
        try:
            pd.DataFrame(columns=['Run ID','Timestamp','Overall Accuracy','Num Agents','Num Obstacles']).to_csv(model.accuracy_file, index=False)
        except Exception as e:
            print("Warning: cannot pre-create accuracy file:", e)

    # monkeypatch model.save_accuracy_record to a safe wrapper if present (so it won't error)
    def safe_save(overall_accuracy):
        path = getattr(model, "accuracy_file", os.path.join(os.getcwd(), f"simulation_accuracy_log_dynamic_run{run_idx+1}.csv"))
        record = {
            'Run ID': getattr(model, "run_id", f"run_{run_idx+1}"),
            'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Overall Accuracy': overall_accuracy,
            'Num Agents': len([a for a in model.schedule.agents if getattr(a,"position",None) is not None or getattr(a,"pos",None) is not None]),
            'Num Obstacles': getattr(model, "num_obstacles", getattr(model, "n_obstacles", N_OBSTACLES))
        }
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                df = pd.read_csv(path)
            else:
                df = pd.DataFrame(columns=record.keys())
        except Exception:
            df = pd.DataFrame(columns=record.keys())
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            print("Warning: could not write accuracy file:", e)
    # apply safe save if model has that method
    if hasattr(model, "save_accuracy_record"):
        model.save_accuracy_record = safe_save

    # Run the model: use stagnation detection as well as "all reached" break
    stagnant_count = 0
    # Snapshot positions for stagnation detection: initialize with current state
    last_positions = []
    for a in model.schedule.agents:
        p = getattr(a, "position", None)
        if p is None:
            p = getattr(a, "pos", None)
        last_positions.append(p)

    for t in range(MAX_STEPS):
        # silence prints from inside model.step() by redirecting stdout temporarily
        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    model.step()
        except Exception as e:
            # If something goes wrong we try once with normal stdout to show error
            print("Exception in model.step() (attempting visible run):", e)
            traceback.print_exc()
            try:
                model.step()
            except Exception:
                pass
            break

        # close any open figures to avoid memory bloat
        try:
            plt.close('all')
        except Exception:
            pass

        # collect current positions (support both pos and position attributes)
        positions = []
        for a in model.schedule.agents:
            p = getattr(a, "position", None)
            if p is None:
                p = getattr(a, "pos", None)
            positions.append(p)

        # compare to last_positions to detect movement
        if positions == last_positions:
            stagnant_count += 1
        else:
            stagnant_count = 0
            last_positions = positions

        # break conditions
        known_positions = [p for p in positions if p is not None]
        if known_positions and all((p == model.goal) for p in known_positions):
            print("All agents at goal — breaking at step", t)
            break
        if stagnant_count >= STAGNATION_WINDOW:
            print(f"No agent movement for {stagnant_count} consecutive steps — breaking (stagnation).")
            break

    # ----- COLLECT DYNAMIC RUN RESULTS -----
    for a in model.schedule.agents:
        aid = getattr(a, "unique_id", None)
        if aid is None:
            continue
        # skip obstacle agents if their ids are in the obstacle range (your Dynamic uses 1000+ for obstacles)
        if isinstance(aid, int) and aid >= 1000:
            continue

        pos = getattr(a, "position", None) or getattr(a, "pos", None)
        # Steps: prefer steps_taken attribute else fallback to steps or len(path_history)-1
        steps = getattr(a, "steps_taken", None)
        if steps is None:
            steps = getattr(a, "steps", None)
        if steps is None:
            hist = getattr(a, "path_history", None) or getattr(a, "path", None)
            steps = max(0, len(hist)-1) if hist else MAX_STEPS

        # path length: prefer actual list lengths
        hist = getattr(a, "path_history", None) or getattr(a, "path", None)
        path_len = len(hist) if hist else (int(steps) + 1)

        try:
            dyn_rows.append({
                "model": "dynamic_hybrid",
                "agent_id": int(aid),
                "reached": bool(pos == model.goal),
                "steps": int(steps),
                "path_length": int(path_len),
                "run": int(run_idx+1)
            })
        except Exception as e:
            print("Warning: failed to append row for agent", aid, ":", e)

    print(f"Collected {len([r for r in dyn_rows if r['run']==run_idx+1])} rows from dynamic run {run_idx+1}.")

# append dynamic rows if present
if dyn_rows:
    all_dfs.append(pd.DataFrame(dyn_rows))

# combine and save
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)

    # Normalize: if 'mode' exists, prefer that for baseline models
    if 'mode' in combined.columns and 'model' in combined.columns:
        # prefer non-null 'model', else use 'mode'
        combined['model'] = combined['model'].fillna(combined['mode'])
        combined = combined.drop(columns=['mode'])

    # Basic cleaning: drop rows that are obviously obstacle placeholders (agent_id >= 1000)
    # (shouldn't be present because we filtered them, but safe to run)
    combined = combined[~(combined['agent_id'].astype(str).str.isnumeric() & (combined['agent_id'].astype(int) >= 1000)) | (combined['model']=='dynamic_hybrid')]

    try:
        combined.to_csv(OUT_CSV, index=False)
        print("\nSaved combined CSV:", OUT_CSV, " shape:", combined.shape)
    except Exception as e:
        print("ERROR writing combined CSV:", e)

    # also write a cleaned CSV where columns are coerced to expected dtypes
    try:
        combined_clean = combined.copy()
        # coerce types
        for c in ['agent_id','steps','path_length','run']:
            if c in combined_clean.columns:
                combined_clean[c] = pd.to_numeric(combined_clean[c], errors='coerce').fillna(0).astype(int)
        if 'reached' in combined_clean.columns:
            combined_clean['reached'] = combined_clean['reached'].astype(bool)
        combined_clean.to_csv(OUT_CSV_CLEAN, index=False)
        print("Saved cleaned CSV:", OUT_CSV_CLEAN, " shape:", combined_clean.shape)
    except Exception as e:
        print("ERROR writing cleaned CSV:", e)

    # produce a quick summary and save bar charts
    try:
        summary = combined_clean.groupby('model').agg(
            Success_Rate = ('reached', lambda s: 100.0 * s.astype(bool).mean()),
            Avg_Steps = ('steps', 'mean'),
            Avg_Path_Length = ('path_length','mean'),
            Count = ('agent_id','count')
        ).round(2).sort_values('Avg_Steps')
        print("\nSummary:\n", summary)

        # save bar charts (PNG) non-interactively
        summary['Success_Rate'].plot(kind='bar', title='Success Rate by Model', ylabel='Success %', xlabel='Model', figsize=(8,4))
        plt.tight_layout()
        plt.savefig(PLOT_PREFIX + "success_rate.png")
        plt.close()

        summary['Avg_Steps'].plot(kind='bar', title='Average Steps by Model', ylabel='Steps', xlabel='Model', figsize=(8,4))
        plt.tight_layout()
        plt.savefig(PLOT_PREFIX + "avg_steps.png")
        plt.close()

        summary['Avg_Path_Length'].plot(kind='bar', title='Avg Path Length by Model', ylabel='Path Length', xlabel='Model', figsize=(8,4))
        plt.tight_layout()
        plt.savefig(PLOT_PREFIX + "avg_path_length.png")
        plt.close()

        print("Saved plots with prefix:", PLOT_PREFIX)
    except Exception as e:
        print("Warning: failed to produce summary or plots:", e)

else:
    print("No data to save (no runs produced output).")

# restore plt.show (safe)
plt.show = _real_show

print("Done.")
