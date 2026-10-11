# Final refactoring plan for dvd_upscale.py

All line numbers below refer to the tagged base, `refactor-base`, which is today's 5772-line file. Lines move after every step, so each step names its code by function name or by a quoted statement, with the line only as a hint.

## 0. Decisions

| Question | Decision |
|---|---|
| Overall shape | minimal-risk's module map and bottom-up move order. Its name dependencies were checked against the code. Two fixes: the resume and fingerprint code goes into a new `resume.py` above `filters.py`, which removes the `workfolder -> filters -> vhs -> workfolder` cycle, and the test tools reuse the existing ones. |
| Package name | `dvd_upscale_parts/`. It cannot shadow `dvd_upscale.py` (a folder called `dvd_upscale/` would), and it tells the user what the folder is. |
| Entry point | `dvd_upscale.py` stays the entry point for `Upscale.bat`, `setup_windows.ps1` (`--faces-worker --check`), the `--faces-worker` child, the `--all`/`--queue` children and `--clip --upscale`. |
| Options object | One argparse Namespace `a`, with every attribute name kept. There is no Settings class, no RunState object and no CONFLICTS engine. The program is made safer with written contracts (DERIVED, RESUME_RULES, CONFLICTS as lists) and with tests. |
| Order of work | Tools and baselines first. Then the launcher. Then pure moves, each proven by line conservation. Then split `main()` in place, then move the pieces. Restructuring threaded code and the detectors is optional and comes last. |
| Test tools | Extend the tracked `dvd_upscale_dev/golden.py`; do not write a new one. Add `msgrun.py` (message texts, gating), `check_moved.py`, `resumecheck.py` and `dvd_upscale_dev/tests/` (stdlib unittest). |

---

## 1. Module map

Layers, lowest first. A module may import only from lower layers (or its own layer where an edge is listed). No module may ever `import dvd_upscale`.

| Layer | Module (≈lines) | Contents (moved word for word unless noted) |
|---|---|---|
| — | `dvd_upscale.py` (≈200) | User docstring (unchanged), stdlib imports, launcher (section 2), re-exports, the `if __name__ == "__main__":` block word for word. During the migration it also holds the code not yet moved. |
| 0 | `__init__.py` (≈15) | Plain-language docstring ("the rest of dvd_upscale.py; start the program with dvd_upscale.py") and `PARTS_VERSION = 1`. It imports nothing. |
| 0 | `common.py` (≈290) | `set_entry`, `script_path()`, `script_dir()` (new). `OUTPUT_TAG`, `STOPPED`. `nostdin` … `hms` (`OUTPUT_TAG` to `def hms`). `flush_to_disk`, `replace_file`, `write_durably`, `keep_awake`, `hold_lock`, `lock_work`, `pngs`, `gap_message`. `class Background`. |
| 1 | `media.py` (≈240) | `MPEG_EXT`, `MKV_SUBS`, `MP4_AUDIO`, `MP4_SUBS` (out of the VHS block, because `probe()` uses `MPEG_EXT`). `probe`, `sample_start`, `detect_mode`, `pulldown_mix`, `cadence`, `video_start`, `count_frames`, `png_size`, `check_ffmpeg`, `probe_or_exit`, `media_duration`, `video_height`, `video_length`, `has_dvd_pcm`. |
| 1 | `workfolder.py` (≈70) | `TYPE_NAMES`, `TYPE_DIRS`, `previous_settings`, `previous_type`. Only `json` and `Path`, so it stays a leaf. |
| 1 | `faces.py` (≈440) | `FACE_*` constants (not `FACE_LOCK` or `FACE_PROCS`). `face_read` … `restore_faces`, `face_dists`, `ORT_DISTS`, `CV_DISTS`, `faces_worker_main`. numpy, cv2 and onnxruntime stay imported inside the functions. |
| 1 | `detect_pictures.py` (≈390), `detect_motion.py` (≈470), `detect_tape.py` (≈455) | All of `_cs_*`, `_ct_*` and `_src_*`, with their prefixes and the `_cs_ABS` and `_ct_NOWIN` caches. Stdlib only. |
| 1 | `outputs.py` (≈110) | `default_output`, `MOVIE_EXT`, `source_id`, `made_from`, `is_upscaled_output`, `WORK_FILES`, `clean_work_folder`. |
| 2 | `vhs.py` (≈260) | `VHS_CADENCE_VF`, `VHS_VIDEO`, `VHS_MODEL_ZIPS`, `_phase_share`, `_low_phases`, `vhs_detect`, `vhs_chroma_delay`, `vhs_geometry`, `vhs_setup`. (Needs media and workfolder.) |
| 2 | `upscaler.py` (≈260) | `gpu_list` … `model_installed`. This is the only home of `AUTO_THREADS`, `LANE_STATUS`, `LANE_PROGRESS`, `UPSCALER_FRAMES`, `UPSCALING`, `DISK_LOCK` and `HandBack`. |
| 2 | `options.py` (≈330) | `check_values`, `build_parser`, `COMMANDS`, `VALUE_OPTS`. New, as documentation: `DERIVED`, `RESUME_RULES`, `CONFLICTS` (section 5). (Needs faces for the `FACE_*` help defaults.) |
| 2 | `faces_setup.py` (≈150) | `face_models_dir`, `pip_cmd`, `face_install_hint`, `face_reinstall_hint`, `faces_check` (`__file__` becomes `script_path()`/`script_dir()`). |
| 2 | `detect.py` (≈180) | `detect_content`, `detect_source`, `resolve_type`. |
| 3 | `filters.py` (≈130) | `vhs_prefilter` stays in vhs.py; filters.py imports it. `prefilter`, `postfilter`, `nvenc_args`, `cpu_args`, `encode_args`, `pick_nvenc`. Later `choose_encoder(a, fps)` (uses `workfolder.previous_settings`). |
| 3 | `stabilize.py` (≈150) | `stab_index`, `write_stab_slice`. Later `setup_stabilize(a, info, fps)` and `measure_shake(a, work, plan, cut, eps, total)`. |
| 4 | `chunk.py` (≈430) | `FACE_LOCK`, `FACE_PROCS`, `class Chunk`, `stop_face_workers` and its `atexit.register`. |
| 5 | `resume.py` (≈200) | Cut from main: `open_work_folder`, `decide_rgb_interp`, `settings_fingerprint` (pure), `settings_change_message` (pure), `check_same_settings`, `save_settings`, `recheck_recent_chunks`. It sits above filters because the fingerprint calls `prefilter(a)`. |
| 5 | `movie_setup.py` (≈480) | Cut from main: the phases before the work folder (section 4). |
| 5 | `scheduler.py` (≈340) | `plan_chunks`, `prepare_chunk_input`, `queue_position`, `upscale_chunks` (the whole chunk loop with its closures). |
| 5 | `finish.py` (≈220) | `join_chunks`, `source_streams`, `audio_options`, `subtitle_options`, `audio_offset`, `write_movie`, `report_length`. |
| 5 | `queue.py` (≈520) | `VIDEO_EXT`, `_WORD`, `queue_parse` … `queue_read`, `queue_finished`, `folder_jobs`, `check_finished_movie`, `to_recycle_bin`, `detected_type`, `queue_main` (`__file__` becomes `script_path()`). |
| 5 | `clip.py` (≈110) | `hms_text`, `clip_main`, `clip_upscale` (`script_path()`). |
| 6 | `movie.py` (≈80) | `main()`: about 45 lines calling the phases in today's order. |

Checked edges: chunk needs `STOPPED`, `gap_message`, `count_frames`, `FACE_FIDELITY`, `write_stab_slice`, the GPU state and the filters, and all of them are moved before chunk.py. queue needs options, outputs, media, workfolder and `STOPPED`. vhs needs only media, workfolder and common. The fingerprint is no longer in workfolder, so there is no cycle.

## 2. Launcher and how the package is found in the embeddable Python

`python312._pth` replaces sys.path. Its `.` entry means the `python\` folder, not the script folder. `setup_windows.ps1` only switches on `import site`. So the launcher must add the script folder itself. This goes right after the docstring and stdlib imports:

```python
_here = os.path.dirname(os.path.realpath(__file__))
if _here not in sys.path:
    sys.path.append(_here)          # append: python\, models\ ... can't shadow real modules
PARTS_VERSION = 1
_BROKEN = ("The dvd_upscale_parts folder next to dvd_upscale.py is missing, incomplete or from "
           "another version: copy dvd_upscale.py and the whole dvd_upscale_parts folder again "
           "(they hold the program together).")
try:
    import dvd_upscale_parts
    from dvd_upscale_parts import common
    ...every other module the launcher re-exports, imported here...
except ImportError as e:
    if (e.name or "").split(".")[0] != "dvd_upscale_parts":
        raise                       # e.g. a real bug: show it as today
    sys.exit(_BROKEN)
if getattr(dvd_upscale_parts, "PARTS_VERSION", None) != PARTS_VERSION:
    sys.exit(_BROKEN)               # stale folder, or one copied without __init__.py
common.set_entry(os.path.realpath(__file__))
```

What this handles:
- `ImportError` with `e.name` covers a missing folder, a missing module file, and a stale module that lacks a name (`cannot import name X from dvd_upscale_parts.y` sets `e.name` to that module).
- A folder without `__init__.py` imports as a namespace package with no `PARTS_VERSION`, which is also covered.
- numpy and cv2 errors have other names and are re-raised.

The message is new text, but it replaces a traceback that can't happen today, so no existing message changes.

`script_path()` and `script_dir()` read the value at call time. The package never captures it with `from common import SCRIPT`. A renamed entry such as `dvd_upscale (1).py` still starts its children on itself, as `Path(__file__)` does today.

All modules are imported eagerly at startup, as today. That keeps `atexit.register(stop_face_workers)` in every process. Lazy dispatch would gain nothing.

Re-exports:
- During the migration, the launcher imports every moved name back, because the code not yet moved needs them.
- At the end it keeps only `__all__ = [prefilter, postfilter, Fraction, VHS_CADENCE_VF, restore_faces, face_eye_dist, main, queue_main, clip_main, faces_worker_main, COMMANDS]`. That list is what `filtertest.py` and `evalfaces.py` use, plus the dispatch targets.

## 3. Global state, circular imports, `__file__`, locks

1. **No `import dvd_upscale` inside the package.** The script runs as `__main__`, so that import would run the script a second time and give duplicate `FACE_PROCS`, `AUTO_THREADS` and lane dicts. A test enforces this. A missing name is fixed by moving it lower, never by importing upward.
2. **Shared mutable state has exactly one home:**
   - upscaler.py: `AUTO_THREADS` (changed in place with `[0] -= 2`), the four lane dicts, `DISK_LOCK`;
   - chunk.py: `FACE_LOCK`, `FACE_PROCS`;
   - the detector modules: `_cs_ABS`, `_ct_NOWIN`, `_cs_SQ`.

   The code has no `global` statements today and only one `nonlocal` (`encoding`). Other modules may use `from .upscaler import LANE_STATUS` because the objects are only changed in place. Tests check that the objects are the same everywhere and that no module rebinds them outside their home (AST).
3. **Circular imports.** The layers are enforced by an AST test (a `LAYERS` dict in `test_structure.py`). Modules move bottom-up, and ruff F82 (undefined name) shows any name that is still left behind.
4. **`__file__`.** Seven uses become `script_path()`/`script_dir()`:
   - `face_models_dir`, `faces_check`, `Chunk.face_worker`, `main` (`here = ...`), `queue_main` (`script, log = ...`), `clip_upscale`;
   - the launcher's `_here` stays as it is.

   `faces_worker_main` keeps `getattr(sys.modules[module], "__file__", None)` for a different purpose. The test allows `__file__` only in common.py and that one string in faces.py.
5. **Locks must stay referenced.**
   - `a.lock = lock_work(work)` stays an attribute of `a`, set inside `open_work_folder`.
   - `qlock` stays a local of `queue_main` for its whole life. If `queue_main` is split, the lock stays in the function that lives as long as the queue does.
   - A test in `test_locks.py` checks that after `open_work_folder` returns, a second `lock_work` on the same folder exits with "Another run is already using…".
6. **Python 3.8.** Gate with `ruff --target-version py38`. Once, after the main split and at the end, run `uv python install 3.8 && uv run -p 3.8 python -m compileall -q dvd_upscale.py dvd_upscale_parts` plus the unit tests. New code must not use `removeprefix`, dict `|`, `list[...]` hints or `match`.

## 4. The main() split (step 20, in place; moved in step 22)

Rules:
- Each new function is a contiguous block of main in its original order.
- Its parameters are the locals the block reads, and it returns the locals later code reads.
- Every change it makes to `a` stays as it is, and every print stays where it is.
- The one early return becomes `if a.analyze: write_analysis_report(a); return`.

```
a = read_options()                       # parse_args, check_values, a.combed, --work default incl. upscale_work fallback (R1)
refuse_output_is_input(a)
find = tool_finder()                     # here = str(script_dir()); check_ffmpeg
resolve_type(a, find)
vhs_info, user = apply_preset(a, find)   # presets dict stays LOCAL; vhs_setup; UserChoices(model, blend, sharpen, chunk, scale)
                                         #   namedtuple; ai_blend/smooth/sharpen re-checks; R5; vhs_info=None unless vhs
check_late_values(a)                     # --telecine->mode, dar, fps, height, chunk_frames (texts unchanged)
choose_output_file(a)                    # default/" test.mkv" name, suffix, longer-preview, someone-else's-file, mkdir
check_tools(a, find)
if not a.fast and not a.analyze:
    choose_model(a, find, user)          # a.esrgan_path; _anime_model_fallback, _vhs_model_fallback, _live_model_fallback,
                                         #   _keep_started_model (R6), _check_model_scale
show_upscaler_settings(a)
drop_unsuitable_faces(a)
info = probe_source(a, vhs_info)         # Source line, matrix, src_hd, pal, already-HD
output_size(a, info)                     # vhs crop/sizes, out_w, height, level, odd-aspect warning
measured = detect_scan_mode(a, info)     # cadence; _report_vhs_mode; _measure_vhs_chroma; _detect_dvd_mode (R4)
if a.analyze: write_analysis_report(a); return
if a.faces: faces_check(a)
fps = choose_fps(a, info, measured)      # R7; a.fps; level 4.2
align_vhs_chunks(a, fps)
if a.stabilize: setup_stabilize(a, info, fps)
choose_encoder(a, fps)                   # gop, R8, NVENC fallback, "Output:" line
work, st = open_work_folder(a)           # mkdir, a.lock (stays on a), keep_awake, stat - same order
decide_rgb_interp(a)                     # R9
fp = settings_fingerprint(a, st)         # pure; same keys, same order, same conditional keys
check_same_settings(a, work, fp)         # damaged-file exit, moved-file tolerance, settings_change_message(...)->str
save_settings(a, work, fp)               # settings.json, detected.json
plan, total, vo, eps, vlen = plan_chunks(a, info, fps)   # pure (Fraction/float mix untouched)
cut = prepare_chunk_input(a, work, vo, vlen)             # a.chunk_input, MPEG stream copy
if a.stabilize: measure_shake(a, work, plan, cut, eps, total)
queue, later = queue_position()          # its own DVD_UPSCALE_QUEUE parse, as today
recheck_recent_chunks(work, plan)        # damaged-chunk check, "Resuming:" line
chunks = upscale_chunks(a, info, work, plan, fps, cut, eps, queue, later)   # step 21
joined = join_chunks(work, chunks, total)
write_movie(a, info, work, joined, total)   # source_streams, audio_options, subtitle_options, audio_offset, run_steps, rename
report_length(a, work, joined, total)
```

The exact return tuples come from the code: ruff F82 names any local that was forgotten.

`upscale_chunks`, first pass: everything from `chunks, notes = {}, []` to the "every chunk must be there" check moves as one block. The closures, `nonlocal encoding`, `gpu=devices[0]` and the try / except BaseException / finally all stay unchanged.

## 5. Settings and conflicts layer (behaviour-preserving)

### DERIVED (options.py)
DERIVED is a commented dict `name -> (phase that sets it, meaning)`. It covers every attribute set on main's Namespace after parsing:
- setup: combed, work, detected, type, denoise, ai_blend, smooth, sharpen, model, scale, chunk_frames, mode, dar, output, esrgan_path, faces;
- source and output: matrix, src_hd, pal, vhs_crop, vhs_sw, vhs_sh, vhs_h, vhs_trim, dvd_trim, out_w, height, level, chroma_shift, mask, vhs_fin, vhs_auto, vhs_dedup, parity, vhs_detail, vhs_warm;
- rate, stabilize and encoder: fps, stab_tf, stab_margin, stab_file, stab_index, gop, cpu, enc, nvenc_error;
- work folder: lock, rgb_interp, chunk_input;
- faces and GPU: face_models_path, face_provider, face_time_shown, gpu_shown;
- chunk loop: stop_lanes, stopping, cancel_lanes, retire_lanes, chunk_label (written, never read: kept).

Test: an AST scan of the package collects `a.X`, `self.a.X`, `getattr(a|self.a, "X", …)` and `setattr`. It skips the functions that use other parsers' Namespaces: `queue_main`, `clip_main`, `clip_upscale`, `faces_worker_main` and its `w`, and `folder_jobs`/`detected_type` on parsed copies. Each name must be a main-parser dest or in DERIVED. This is union semantics: an attribute set only on some paths is still allowed.

### RESUME_RULES (options.py comment plus tests)
These are the places where an earlier run wins:

| Rule | What wins | Where |
|---|---|---|
| R1 | the old shared `upscale_work` folder | `read_options` |
| R2 | type | `previous_type` |
| R3 | VHS mode | `vhs_setup` |
| R4 | DVD mode | `_detect_dvd_mode` |
| R5 | chunk size when model and scale are the same | `apply_preset` |
| R6 | substitute model with its blend, sharpen and chunk | `_keep_started_model` |
| R7a / R7b | fps when automatic / when given | `choose_fps` |
| R8 | CPU encoder | `choose_encoder` |
| R9 | rgb interpolation off | `decide_rgb_interp` |

`previous_settings()` stays uncached at all 10 call sites. That is equivalent, because nothing writes settings.json before `save_settings`, but caching would gain nothing.

### CONFLICTS (options.py comment table; every row gets a msgrun case; no engine)
Each row lists: condition, the function that handles it, today's outcome.

**Errors today (EXIT, texts unchanged):**
1. Bad values (check_values, and the re-checks after the preset).
2. Output == input.
3. Output not .mkv/.mp4/.m4v.
4. `--test` would replace a longer existing movie.
5. The output exists and was not made by this script, or was made from another source.
6. A tool is missing.
7. A model is missing; x4plus without `--scale 4`; no model for VHS.
8. The source is already HD (only a NOTE under `--analyze`).
9. `--stabilize` without vid.stab.
10. `--work` with `--all`.
11. The resume fingerprint changed ("Settings or input changed" with its hints).
12. settings.json is damaged.

**Notes or overrides today (kept):**
13. `--telecine` sets mode=telecine (silent).
14. `--mask`/`--chroma-delay` with an automatic type mean vhs ("Type: VHS tape (… given …)"), except in `--all` folders.
15. A type folder under `--all`.
16. `--faces` is dropped with a NOTE for `--fast`, anime/cgi or `--ai-blend 0` (silent with `--analyze`).
17. The faces strength is capped by ai_blend (faces_check text).
18. `--stabilize` on a non-tape source (NOTE).
19. Odd aspect (WARNING).
20. VHS chunk rounding (NOTE).
21. Model fallback NOTEs; "Keeping …" for a resumed model.
22. x4 chunk cap at 480 and the x4plus blend/sharpen defaults (silent).

**Silent today, candidates for a NOTE (optional step O5, needs the user's OK):**
- `--fast` ignores `--model`, `--scale`, `--tile`, `--gpu-threads`, `--ai-blend`, `--smooth` and `--gpu-jobs`;
- `--mask`/`--chroma-delay` are ignored with an explicit non-tape type;
- `--face-model`/`--face-models` without `--faces`;
- `--delete-originals` with `--test` is skipped.

**VALUE_OPTS** stays next to build_parser. A test checks that it equals the option strings whose action takes exactly one value (`nargs is None` and not a flag). The `--stabilize`/`--faces` gap is in the quirk list (section 9).

## 6. Other long functions (optional steps O1–O4, one per commit, each moved word for word)

- `queue_main`, split into:
  - `_queue_parser`, `_folder_setup`, `_queue_file_setup`, `QueueLog`;
  - a module-level `delete_original`, `_movies_after`, `_run_movie` (with the Ctrl+C wait), `_shutdown`.

  `qlock` and the tried/done/failed sets stay in `queue_main`.
- `run_upscaler`: `_watch_upscaler` (with its finally), `_show_gpus_once`, `_upscaler_failed`.
- `faces_worker_main`: `_worker_parser`, `_watch_parent`, `_check_packages`, `_check_opencv`, `_check_model_files`, `_self_test`, `_restore`. Exit codes and keywords stay the same.
- `faces_check`: `_run_check_worker`, plus pure message builders on `(rc, found, lines)`.
- `Chunk.upscale`: `_reserve_disk`, `_upscale_with_retry`, `_encode_command`. `Chunk.__init__`: `_source_args`, still under `make_lock` and still writing the trims.
- `build_parser` is **not** split: it is a flat list, and splitting it risks `--help`.
- Detectors (O6, last; only with byte-identical full detection dicts): `_src_detect_source`, `_cs_detect_content`. Floating-point sums are never reordered.

## 7. Test suite and tools

### Tools in `dvd_upscale_dev/` (all stdlib)
- **golden.py (extended in step 0):**
  - (a) Messages gate: a scenario fails when `Counter(messages)` differ. Today `gate` is computed before the message check.
  - (b) `compare` only checks scenarios present in NEW and lists the rest as "not run".
  - (c) `GOLDEN_SCRIPT` env var overrides `SCRIPT`.
- **msgrun.py `record TAG` / `compare A B` (new, gating, exact order):** about 50 commands run as `[sys.executable, "-I", SCRIPT, ...]` from a temporary cwd, with `COLUMNS=100` (help also at 80). It records rc, stdout and stderr with paths normalised. Cases:
  - `--help`, `--commands`, no arguments, `--queue --help`, `--all --help`, `--clip --help`, `--faces-worker --help`;
  - `--faces-worker --check --models <empty>`;
  - every check_values error (`--ai-blend 2`/`nan`, `--smooth -1`, `--sharpen 3`, `--dar 0:1`, `--fps 0`, `--height 7`, `--chunk-frames 0`, `--test -1`, `--gpu-threads 17`, `--gpu-jobs 5`, `--faces 0`, `--mask 1:2`, `--chroma-delay x`);
  - CONFLICTS rows 2–10 and 13–21;
  - the model fallbacks, using a copy of `fakeesrgan` in a temp folder whose `models/` holds only the chosen `.param`/`.bin` files, run as `--test 1 --height 240 --cpu` on `sync/cut.mpg`: anime with only x2plus, anime with only x4plus, VHS camcorder without dn50, VHS with no model, live without x2plus, an explicit missing `--model`, x4plus with `--scale 2`;
  - resume hints: run, then re-run with `--hevc`, with `--cpu` dropped, with another `--mode`, with `--face-model codeformer`;
  - the missing/incomplete/stale-folder messages, using a temp copy of the install (skipped before step 1).

  Run with the `SCRIPT` env var to use the base copy.
- **check_moved.py `[REV=HEAD]` (new):** reads `dvd_upscale.py` plus `dvd_upscale_parts/*.py` at REV and in the working tree. It removes blank lines, import statements (by AST span) and module docstrings, then compares the `Counter`s of stripped lines. It prints removed and added lines and exits 0 only if both are empty, or if they match `--allow FILE` (a list of expected edits, such as the `__file__` lines).
- **resumecheck.py OLD NEW (new):**
  - For live_ai, vhs_ai, stabilize_ai and (if face models exist) faces_ai, from golden's SCENARIOS: run OLD to the end, delete the output and the 2 newest `chunk_*.mkv`, run NEW in the same work folder.
  - Expect rc 0, `Resuming: N-2 of N`, no "Settings or input changed", and framemd5 equal to OLD's output.
  - Then the same with NEW and OLD swapped.
- **Base copy:** `git show refactor-base:dvd_upscale.py > dvd_upscale_dev/base/dvd_upscale.py` (gitignored). The base is a single file, so this copy is complete.

### Unit tests in `dvd_upscale_dev/tests/` (run: `cd /home/user/My-Codes && python3 -m unittest discover -s dvd_upscale_dev/tests -q`)
- **support.py:**
  - puts REPO on sys.path;
  - provides `du`, a merged read-only view of the names in `dvd_upscale` and every existing `dvd_upscale_parts` module, so tests don't change as code moves (patching is done on the home module, found with `home(name)`);
  - `ns(**kw)` = `build_parser()` defaults for `["in.mkv"]` plus neutral DERIVED values;
  - helpers `tmp_work(settings, detected, input_size_mtime)` and `fake_esrgan(models)`;
  - skip decorators `needs_numpy` and `needs_ffmpeg`;
  - snapshot load and compare.
- **make_snapshots.py:** run once in step 0 against `base/dvd_upscale.py`, loaded by path (as evalfaces does). It writes `tests/snapshots/*.json`:
  - prefilter, postfilter and vhs_prefilter for filtertest's cases plus dar, HD, PAL, warm trims and the stabilize tail;
  - nvenc/cpu/encode args for hevc × level × gop;
  - esrgan_cmd and gpu_threads (compact/big, tile, gpu);
  - default_output;
  - queue_parse;
  - stab_index on generated text and binary TRF files.
- **First tests (step 0, passing on the unchanged file):**
  - `test_filters.py`, `test_upscaler.py` (`_read_log_updates` with a half-written line, `check_frames` on hand-made PNG headers, `AUTO_THREADS` lowering through `gpu_threads`);
  - `test_options.py` (check_values exact texts via `SystemExit.code`, VALUE_OPTS, DERIVED contract, default_output);
  - `test_queue.py` (queue_parse quotes, `''`, `#`, unclosed quote; queue_lines BOMs; queue_test_secs; queue_read in a temp folder);
  - `test_resume.py` (previous_settings: size/mtime mismatch, damaged, missing; previous_type order: detected.json, the "vhs" key, denoise, model);
  - `test_stab.py`;
  - `test_faces.py` (face_tracks, face_plan, skipUnless numpy).
- **Added with each step:**
  - `test_structure.py` (from step 1): each module imports alone via `python3 -I -c "import sys; sys.path.append(REPO); import dvd_upscale_parts.X"` (plain `-I -c` from the repo root does not see the cwd); LAYERS; no `import dvd_upscale`; shared-state identity and no-rebind; the `__file__` whitelist; the re-export list; `script_path()` ends in `dvd_upscale.py`; `PARTS_VERSION` matches.
  - `test_locks.py` (step 20h).
  - `test_resume.py` additions (step 20h): `settings_fingerprint` equals the golden base settings.json for every golden scenario (input path, size and mtime normalised), and old layouts without "rgb"/"stabilize" from `runs/` copied into `tests/fixtures/`; `settings_change_message` texts; R1–R9 each on a temp work folder.
  - `test_plan.py` (step 20i): plan_chunks (exact multiple, remainder, `--test` cap, zero-frame chunks dropped); choose_fps (NTSC/PAL telecine, VHS interlaced ×2, a typed 23.976, kept fps); output_size (1920 cap, levels).
  - `test_models.py` (step 20d): the fallbacks with `fake_esrgan`.
  - `test_finish.py` (step 20j): audio_options and subtitle_options (mkv/mp4, pcm_dvd to FLAC, mov_text to SRT, dvd_subtitle in mp4), audio_offset.

### Gates (exact commands)

**GATE-MOVE** runs after every step:
```
cd /home/user/My-Codes
ruff check --target-version py38 --select E9,F63,F7,F82 dvd_upscale.py dvd_upscale_parts
python3 dvd_upscale_dev/check_moved.py HEAD          # "moved cleanly" (or only the step's --allow list)
python3 -m unittest discover -s dvd_upscale_dev/tests -q
cd dvd_upscale_dev
python3 msgrun.py record cur && python3 msgrun.py compare base cur
python3 golden.py record cur live_ai live_two_gpus vhs_ai ntsc_film_ai pal_film_ai stabilize_strong_fast detect_auto detect_auto_vhs clip queue && python3 golden.py compare base cur
PYTHONPATH=.. python3 filtertest.py
```

**GATE-MILESTONE** is GATE-MOVE plus:
```
cd /home/user/My-Codes/dvd_upscale_dev
python3 golden.py record full && python3 golden.py compare base full        # all 20 incl. faces_ai, all_folder
python3 e2e.py ../dvd_upscale.py cur61 > e2e_cur61.txt && diff <(grep -E 'breaks|offset' e2e_base61.txt) <(grep -E 'breaks|offset' e2e_cur61.txt)
PATH=dl/ffmpeg-master-latest-linux64-gpl/bin:$PATH python3 e2e.py ../dvd_upscale.py curm > e2e_curm.txt && (same diff vs e2e_basem.txt)
python3 facerun.py medium > facerun_cur.txt && diff facerun_base.txt facerun_cur.txt
python3 stabrun.py cur 4 > stab_cur.txt && diff stab_base.txt stab_cur.txt
bash simgpu/run.sh > sim_cur.txt; bash simgpu/runR.sh >> sim_cur.txt     # same rc and NOTE lines as base (wall times differ)
FAIL_CHUNKS=tmp_00001,tmp_00003 python3 ../dvd_upscale.py sync/cut.mpg runs/fe/out.mkv --cpu --type live --mode progressive --chunk-frames 30 --test 4 --height 240 --esrgan failesrgan/realesrgan-ncnn-vulkan --work runs/fe/work   # retry notes and AUTO_THREADS 8->6, as base
python3 resumecheck.py base/dvd_upscale.py ../dvd_upscale.py
```
Expected e2e results: `breaks 0` everywhere except `ntsc_film_bff` (one repeated first frame), and AVI offsets of -33/-67 ms. These must match the base logs.

## 8. Steps (one agent per step, in order; commit after each green gate)

| # | Edit | Verify | Risk |
|---|---|---|---|
| 0 | Tools only:<br>• golden.py changes (a–c)<br>• msgrun.py, check_moved.py, resumecheck.py<br>• tests/ with support.py, make_snapshots.py and the first tests<br>• `.gitignore` for `dvd_upscale_dev/base/`<br>Then: make the base copy; `python3 tests/make_snapshots.py`; `golden.py record base`, `record base2` (must PASS against each other, messages included); `msgrun.py record base` twice, identical; e2e base61/basem logs; facerun_base, stab_base, sim_base. Tag `refactor-base`. | All suites give today's results. The unit tests pass on the unchanged file. Breaking one filter string or message by hand makes them fail. | None for the product. If stabilize frames still differ between base and base2 (they did before 2ad82d1), fix the nondeterminism now, or gate those two scenarios on commands only and note it. |
| 1 | Add `__init__.py` and `common.py` holding only `set_entry`, `script_path` and `script_dir`. Launcher as in section 2. README "Put these files together" list: add "the dvd_upscale_parts folder", and say to copy the whole folder when updating. | GATE-MOVE plus `test_structure.py`. Also `python3 -I /home/user/My-Codes/dvd_upscale.py --commands` from `/tmp`. msgrun's broken-folder cases (missing folder; folder without `__init__.py`; a `PARTS_VERSION` changed) give the message and exit 1. **Windows check 1 (recommended):** copy `dvd_upscale.py` and the folder into the user's setup.bat folder and run `python\python.exe dvd_upscale.py --commands`. | Medium: this is the only step about how Python finds code. |
| 2 | Move to common: `OUTPUT_TAG`, the `nostdin`…`hms` helpers, `flush_to_disk`…`pngs`, `gap_message`, `Background`, `STOPPED`. Import them back. | GATE-MOVE | Low |
| 3 | `media.py` (incl. the 4 extension constants, `png_size`, `has_dvd_pcm`) | GATE-MOVE; detect_auto goldens identical | Low |
| 4 | `workfolder.py` | GATE-MOVE | Low |
| 5 | `vhs.py` | GATE-MOVE (vhs_ai, detect_auto_vhs) | Low |
| 6 | `filters.py` | GATE-MOVE; filter snapshots | Low |
| 7 | `upscaler.py` | GATE-MOVE, simgpu run.sh/runR.sh, the failesrgan command | Medium-low (shared state; identity tests) |
| 8 | `stabilize.py` | GATE-MOVE, stabrun | Low |
| 9 | `faces.py` | GATE-MOVE, facerun, `cd dvd_upscale_dev && python3 evalfaces.py` (if its inputs exist) | Low |
| 10 | `faces_setup.py`, `__file__` becomes `script_path()`/`script_dir()` | GATE-MOVE with `--allow` listing the 2 lines; facerun with and without `--face-models` | Medium-low |
| 11 | `chunk.py` (`FACE_LOCK`, `FACE_PROCS`, `Chunk`, `stop_face_workers` and atexit), `__file__` becomes `script_path()` | **GATE-MILESTONE**, plus Ctrl+C: SIGINT to live_two_gpus after 10 s leaves no upscaler or `--faces-worker` process (`pgrep -f`), and a re-run resumes | Medium |
| 12–14 | `detect_pictures.py`, `detect_motion.py`, `detect_tape.py`, one per step | GATE-MOVE, plus a full detection-dict regression: `tests/test_detect_regress.py` compares `detect_content`/`detect_source` on every file under sync/, tc/, pal/, avi/ and shaky.mpg against a snapshot, removing `seconds` at the top level and inside `scores` | Low |
| 15 | `detect.py` | GATE-MOVE, detection regression | Low |
| 16 | `outputs.py` | GATE-MOVE | Low |
| 17 | `options.py`, plus the DERIVED, RESUME_RULES and CONFLICTS tables as comments/data | GATE-MOVE; help goldens byte-identical | Low |
| 18 | `queue.py` (`script_path()`) | **GATE-MILESTONE** (queue, all_folder); `--all --delete-originals` on a temp copy gives log lines equal to base | Medium-low |
| 19 | `clip.py` (`script_path()`) | GATE-MOVE (clip) | Low |
| 20a–j | Split main **in place**, one letter per commit, exactly as listed in section 4:<br>(a) read_options, refuse_output_is_input, tool_finder<br>(b) apply_preset, UserChoices, check_late_values<br>(c) choose_output_file, check_tools<br>(d) choose_model and its 5 helpers<br>(e) show_upscaler_settings, drop_unsuitable_faces, probe_source, output_size<br>(f) detect_scan_mode and its 3 helpers, write_analysis_report<br>(g) choose_fps, align_vhs_chunks, setup_stabilize, choose_encoder<br>(h) open_work_folder, decide_rgb_interp, settings_fingerprint, check_same_settings, settings_change_message, save_settings<br>(i) plan_chunks, prepare_chunk_input, measure_shake, queue_position, recheck_recent_chunks<br>(j) join_chunks, write_movie and its 4 helpers, report_length | Each commit: GATE-MOVE (check_moved: added lines are only `def`, `return`, the call line and namedtuple lines, and it removes nothing else). New unit tests: after (d) test_models; after (h) test_resume additions, test_locks and resumecheck; after (i) test_plan; after (j) test_finish and **GATE-MILESTONE**, plus the Python 3.8 run. | Medium: locals become parameters. ruff F82 catches a missing one; golden commands and the settings files catch a wrong value. |
| 21 | Chunk loop into `upscale_chunks(...)` in place, closures unchanged | **GATE-MILESTONE**, plus the Ctrl+C test twice | Medium (threads, but the block is moved, not rewritten) |
| 22 | Move the phases out, one module per commit: resume.py, movie_setup.py, `filters.choose_encoder`, `stabilize.setup_stabilize`/`measure_shake`, scheduler.py, finish.py, then movie.py (main). Cut the launcher's re-exports down to `__all__`. Run `pyflakes` on all files with no warnings left. | GATE-MOVE each; **GATE-MILESTONE** after the last | Low |
| 23 | Docs: PROGRESS.md gets the module map, the gates and how to run the unit tests. README check. **Windows check 2:** setup.bat (its `--faces-worker --check` runs in the embeddable Python), Preview.bat on one movie, Upscale.bat with `--faces --test 10`, and resuming a work folder started by the old version. | The user confirms all four. | Low |

Optional, after step 23, each its own commit with the gate for its area. Do them only if wanted:
- **O1–O4:** the splits in section 6.
- **O5:** NOTEs for the silent CONFLICTS rows. This changes messages, so it needs the user's OK; the new msgrun goldens are reviewed.
- **O6:** detector splits.
- **O7:** turn `upscale_chunks` into a `ChunkScheduler` class (closures become methods, `nonlocal encoding` becomes `self.encoding`). Thresholds become module constants plus a fake-job unit test. Verify with GATE-MILESTONE run twice.
- **O8:** `prefilter(a, trim=None)`, which falls back to `a.vhs_trim`/`a.dvd_trim`, so filtertest still works. This removes the trim writes guarded by `make_lock`.
- **O9:** a pure `pick_model(..., installed=callable)`.
- **O10:** fix the quirks in section 9 with the user's OK.

## 9. Deliberately NOT changed

- Every message text.
- The `--help` and `--commands` output.
- settings.json and detected.json keys, order and value types.
- The `DVD_UPSCALE_QUEUE`, `_REPORT` and `_CLIP_TEST` protocols, including the three different parses of `DVD_UPSCALE_QUEUE`.
- The face-worker exit codes and keywords.
- Child command lines: `[sys.executable, script_path(), ...]`.
- The Namespace `a` and all its attribute names. No Settings, RunState, freeze or CONFLICTS engine.
- `previous_settings` stays uncached.
- The presets dict stays local to `apply_preset`, because `presets["vhs"]` is replaced for camcorder tapes and repeated calls in one test process would see that change.
- The trims written on `a` under `make_lock` (until O8).
- `AUTO_THREADS` stays a module-level list.
- `build_parser` stays one function.
- The Fraction/float mix in the timing code.
- The detector arithmetic.
- The eager imports at startup.
- The `__main__` block.

**Quirks kept and listed for the user to decide (O10):**
1. `VALUE_OPTS` lacks `--stabilize` and `--faces`, so `--all --stabilize strong` takes a folder named "strong" as the movies folder.
2. `clip_upscale`: `except (OSError, ValueError): rc, found = rc or 1, None` raises UnboundLocalError when `subprocess.run` itself fails.
3. main reuses the name `queue` for a string and later a dict.
4. The re-checks after the preset duplicate check_values and cannot fire for preset values.
5. `a.chunk_label` is written and never read.
6. `--mask`, `--gpu-threads` and `--face-model` are silently ignored in the cases listed in section 5.

## 10. Risks and their guards

- **Embeddable sys.path:** the launcher append, `-I` in msgrun and test_structure, and Windows checks after steps 1 and 23.
- **Partial or stale folder:** the `ImportError` name check plus the mandatory `PARTS_VERSION`.
- **Duplicate globals:** the no-import-of-dvd_upscale test, plus the identity and no-rebind tests.
- **`__file__`:** call-time `script_path()` and the whitelist test.
- **Lock lifetime:** `test_locks.py`.
- **Resume compatibility:** fingerprint golden tests and resumecheck in both directions.
- **Threads:** the block is moved whole, with milestone simgpu, failesrgan and Ctrl+C runs; the class conversion is optional.
- **Detector floats:** moves only until O6, plus the full-dict regression.
- **Not testable on Linux** (msvcrt locking, Recycle Bin, SetThreadExecutionState, NVENC, DirectML): moved word for word and exercised in Windows check 2.

Files: `/home/user/My-Codes/dvd_upscale.py`, `/home/user/My-Codes/dvd_upscale_dev/golden.py` (to extend), `/home/user/My-Codes/dvd_upscale_dev/e2e.py`, `/home/user/My-Codes/dvd_upscale_dev/filtertest.py`, `/home/user/My-Codes/dvd_upscale_dev/evalfaces.py`, `/home/user/My-Codes/setup_windows.ps1`, `/home/user/My-Codes/README.txt`, `/home/user/My-Codes/dvd_upscale_dev/PROGRESS.md`