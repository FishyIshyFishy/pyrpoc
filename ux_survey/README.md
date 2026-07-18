# pyrpoc UX survey

A tiny, self-contained web survey for gathering UX preferences. Participants
work through a series of **A/B cases** (pick the option they'd rather use), then
**download a JSON results file** that you collect and aggregate. No server, no
build step, no internet — it runs by opening one HTML file.

## Files

| File | What it is |
|---|---|
| `index.html` | The page shell. Open this to take the survey. |
| `survey.css` | Styling (light/dark aware, responsive A/B cards). |
| `survey.js` | The engine: renders cases, records choices, downloads results. **You don't edit this.** |
| `cases.js` | **The content you edit** — survey config + the list of A/B cases. |
| `build.py` | Inlines everything into a single `survey.html` for distribution. |
| `survey.html` | Generated one-file survey (run `build.py`). **Send this to participants.** |
| `aggregate.py` | Optional: tally collected result files into a summary + CSV. |
| `results/` | Drop collected participant `.json` files here for `aggregate.py`. |

## Adding cases

Everything lives in **`cases.js`**. Add entries to `SURVEY_CASES`; the schema is
documented at the top of that file. Minimal shape:

```js
{
  id: "start-stop-model",
  operation: "Starting acquisition",
  prompt: "Which way of starting imaging do you prefer?",
  options: [
    { id: "a", label: "Two buttons", description: "Separate Snap and Live.",
      previewText: "[ > Snap ]  [ >> Live ]  [ [] Stop ]" },
    { id: "b", label: "One smart button", description: "Start morphs into Stop.",
      previewText: "idle:  [ > Start ]\nrunning: [ [] Stop ]" }
  ]
}
```

Each option can carry a preview in one of three forms (all optional):
- `previewText` — a monospace/ASCII mockup (rendered in a `<pre>` box),
- `image` — a relative path (e.g. `"img/foo.png"`) or a `data:` URI,
- `preview` — arbitrary HTML you author.

Keep each `id` **stable and unique** — results from different people are matched
by `caseId`, so renaming an id after collecting responses will split the data.

Config knobs (top of `cases.js`): `shuffleOptionOrder` (randomize left/right per
participant), `allowComments` (optional "why?" per case), `requireChoice`,
`collectParticipant` (name/role fields), and `returnTo` (the "send it back to…"
instructions shown at the end).

## Running / distributing

- **To try it while editing:** double-click `index.html` (loads the separate
  `survey.css` / `cases.js` / `survey.js` next to it).
- **To distribute (recommended):** run `python build.py` to generate a single
  self-contained `survey.html`, then send *that one file*. Everything (styles +
  engine + cases) is inlined, so a colleague just double-clicks it — no folder,
  no zip-extract step, nothing to install. Works equally well emailed, dropped
  on a shared drive, or hosted on any static host.
  - Why not zip the folder? If a colleague opens `index.html` from *inside* the
    zip without extracting, the browser can't find `survey.css` / `survey.js`
    and shows a broken, unstyled page. The single file avoids that entirely.
- Rebuild `survey.html` (`python build.py`) whenever you change the cases.
- Progress is saved in the participant's browser, so a refresh won't lose work.

## Collecting + aggregating

1. Each participant finishes and clicks **Download my results** → they get
   `ux-survey_<id>_<name>_<timestamp>.json`.
2. They send it back (see `returnTo` in `cases.js`).
3. Put all the collected `.json` files into `results/`.
4. Run `python aggregate.py` → prints a per-case tally with comments and writes
   `summary.csv` (one row per option with counts + percentages).

## Notes

- The scaffold ships with **one placeholder case** so the page renders. Replace
  `SURVEY_CASES` in `cases.js` with your real cases and delete the example.
- Uses classic `<script>` tags and globals (not ES modules) specifically so it
  works from `file://` without a server.
