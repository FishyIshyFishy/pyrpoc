/*
 * cases.js — survey configuration + the A/B cases.
 *
 * This file is loaded as a plain <script> (no modules), so it just assigns two
 * globals: SURVEY_CONFIG and SURVEY_CASES. Edit this file to add your cases;
 * you do not need to touch survey.js / index.html.
 *
 * ---------------------------------------------------------------------------
 * CASE SCHEMA (each entry in SURVEY_CASES)
 * ---------------------------------------------------------------------------
 * {
 *   id:        "unique-stable-id",   // REQUIRED, unique; used as the key in results. Do not reuse.
 *   operation: "Short label",         // the operation being compared (shown as a chip)
 *   prompt:    "Which feels better?", // the question the participant answers
 *   context:   "optional sentence",   // optional extra explanation shown under the prompt
 *   options: [                        // 2+ options (A/B = 2). Rendered as side-by-side cards.
 *     {
 *       id:          "a",                  // REQUIRED, unique within this case
 *       label:       "Option A",           // short title
 *       description: "one-line summary",   // optional
 *       // Provide AT MOST one preview form per option (all optional):
 *       previewText: "ASCII / monospace mockup\n...",   // rendered in a <pre> box
 *       preview:     "<div>arbitrary HTML mockup</div>", // rendered as HTML (you author it; trusted)
 *       image:       "img/foo.png",                       // relative path or data: URI
 *     },
 *     { id: "b", label: "Option B", description: "...", previewText: "..." }
 *   ]
 * }
 *
 * Notes:
 *  - Keep `id`s stable across edits so results from different people line up.
 *  - `SURVEY_CONFIG.shuffleOptionOrder` randomizes left/right per participant to
 *    reduce position bias; the presented order is recorded in the results.
 */

window.SURVEY_CONFIG = {
  surveyId: "pyrpoc-ux-v1",          // bump when you change the case set meaningfully
  schemaVersion: 1,                   // results file format version (leave as-is)
  title: "pyrpoc UX preferences",
  intro:
    "Thanks for helping shape the new pyrpoc interface. You'll see a series of " +
    "A/B choices about how common operations could work. For each one, pick the " +
    "option you'd prefer to use — there are no wrong answers, go with your gut. " +
    "At the end you'll download a small file; please send it back so your input " +
    "can be folded into the design.",
  collectParticipant: true,           // show optional name/role fields at the start
  shuffleOptionOrder: true,           // randomize option positions per participant
  allowComments: true,                // optional "why?" note under each case
  requireChoice: true,                // must choose before advancing
  returnTo:
    "Email the downloaded file to ishaan.singh@manasai.co (or send it however " +
    "you got this survey).",
};

/*
 * SURVEY_CASES — REPLACE the single placeholder below with your real A/B cases.
 * The scaffold ships with one obvious example so the page renders and you can
 * see the format. Delete it once you add real cases.
 */
window.SURVEY_CASES = [
  {
    id: "example-placeholder",
    operation: "EXAMPLE — replace me",
    prompt: "This is a placeholder case so the survey renders. Which mockup do you prefer?",
    context:
      "Real cases go in cases.js (see the schema comment at the top of this file). " +
      "Delete this example once you add your own.",
    options: [
      {
        id: "a",
        label: "Option A",
        description: "A short description of what this choice means.",
        previewText:
          "+---------------------------+\n" +
          "|  [ Start ]   Status: idle |\n" +
          "|                           |\n" +
          "|      ( example A )        |\n" +
          "+---------------------------+",
      },
      {
        id: "b",
        label: "Option B",
        description: "A short description of the alternative.",
        previewText:
          "+---------------------------+\n" +
          "|  > Snap   >> Live   [] X   |\n" +
          "|                           |\n" +
          "|      ( example B )        |\n" +
          "+---------------------------+",
      },
    ],
  },
];
