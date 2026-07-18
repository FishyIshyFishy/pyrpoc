/*
 * survey.js — the survey engine. No dependencies, no build step, no network.
 * Reads the globals defined in cases.js (SURVEY_CONFIG, SURVEY_CASES), renders
 * one A/B case at a time, records choices, and lets the participant download a
 * JSON results file at the end. You should not need to edit this file to add
 * cases — edit cases.js instead.
 */
(function () {
  "use strict";

  var DEFAULTS = {
    surveyId: "survey",
    schemaVersion: 1,
    title: "Survey",
    intro: "",
    collectParticipant: true,
    shuffleOptionOrder: true,
    allowComments: true,
    requireChoice: true,
    returnTo: "",
  };

  var CONFIG = Object.assign({}, DEFAULTS, window.SURVEY_CONFIG || {});
  var CASES = Array.isArray(window.SURVEY_CASES) ? window.SURVEY_CASES : [];
  var STORAGE_KEY = "uxsurvey:" + CONFIG.surveyId;

  var app = document.getElementById("app");
  var brandEl = document.getElementById("brand-title");
  var progressEl = document.getElementById("progress");
  var progressLabel = document.getElementById("progress-label");
  var progressFill = document.getElementById("progress-fill");
  var footerNote = document.getElementById("footer-note");

  var state = {
    screen: "intro", // "intro" | "case" | "done"
    index: 0,
    participant: { name: "", role: "", email: "", notes: "" },
    responses: {}, // caseId -> { choiceId, comment, dwellMs }
    order: {}, // caseId -> [optionId, ...] presented order
    startedAt: null,
    finishedAt: null,
    caseShownAt: null,
  };

  // ----------------------------------------------------------------- helpers
  function el(tag, props, children) {
    var node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        if (k === "class") node.className = props[k];
        else if (k === "text") node.textContent = props[k];
        else if (k === "html") node.innerHTML = props[k];
        else if (k === "onclick") node.addEventListener("click", props[k]);
        else if (k === "oninput") node.addEventListener("input", props[k]);
        else if (k in node) node[k] = props[k];
        else node.setAttribute(k, props[k]);
      });
    }
    (children || []).forEach(function (c) {
      if (c == null) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function shuffle(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    return a;
  }

  function slug(s) {
    return String(s || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "anon";
  }

  function loadSaved() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var saved = JSON.parse(raw);
      if (saved && typeof saved === "object") {
        state.participant = Object.assign(state.participant, saved.participant || {});
        state.responses = saved.responses || {};
        state.order = saved.order || {};
        state.startedAt = saved.startedAt || null;
      }
    } catch (e) {
      /* localStorage unavailable (some file:// contexts) — resume is best-effort */
    }
  }

  function save() {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          participant: state.participant,
          responses: state.responses,
          order: state.order,
          startedAt: state.startedAt,
        })
      );
    } catch (e) {
      /* ignore */
    }
  }

  function clearSaved() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
  }

  function presentedOrder(c) {
    if (state.order[c.id]) return state.order[c.id];
    var ids = (c.options || []).map(function (o) {
      return o.id;
    });
    if (CONFIG.shuffleOptionOrder) ids = shuffle(ids);
    state.order[c.id] = ids;
    return ids;
  }

  function stampDwell() {
    if (state.screen === "case" && state.caseShownAt != null && CASES[state.index]) {
      var c = CASES[state.index];
      var r = state.responses[c.id] || (state.responses[c.id] = {});
      r.dwellMs = (r.dwellMs || 0) + (Date.now() - state.caseShownAt);
      state.caseShownAt = null;
    }
  }

  function answeredCount() {
    return CASES.filter(function (c) {
      return state.responses[c.id] && state.responses[c.id].choiceId;
    }).length;
  }

  // --------------------------------------------------------------- rendering
  function setProgress() {
    if (state.screen === "case" && CASES.length) {
      progressEl.hidden = false;
      progressLabel.textContent = "Case " + (state.index + 1) + " of " + CASES.length;
      progressFill.style.width = ((state.index + 1) / CASES.length) * 100 + "%";
    } else if (state.screen === "done") {
      progressEl.hidden = false;
      progressLabel.textContent = "Complete";
      progressFill.style.width = "100%";
    } else {
      progressEl.hidden = true;
    }
  }

  function render() {
    brandEl.textContent = CONFIG.title || "Survey";
    app.innerHTML = "";
    if (state.screen === "intro") renderIntro();
    else if (state.screen === "case") renderCase();
    else if (state.screen === "done") renderDone();
    setProgress();
    window.scrollTo(0, 0);
  }

  function renderIntro() {
    var children = [
      el("h1", { text: CONFIG.title || "Survey" }),
      el("p", { text: CONFIG.intro || "" }),
    ];

    if (CONFIG.collectParticipant) {
      var mk = function (key, label, type) {
        var input = el("input", {
          type: type || "text",
          value: state.participant[key] || "",
          oninput: function (e) {
            state.participant[key] = e.target.value;
            save();
          },
        });
        return el("div", { class: "field" }, [el("label", { text: label }), input]);
      };
      children.push(mk("name", "Name (optional)"));
      children.push(mk("role", "Role / how you use the microscope (optional)"));
      children.push(mk("email", "Email (optional)", "email"));
    }

    if (!CASES.length) {
      children.push(
        el("div", { class: "empty", html: "No cases defined yet. Add them in <code>cases.js</code> (see the schema comment at the top of that file)." })
      );
    }

    var startBtn = el("button", {
      class: "btn btn-primary",
      text: CASES.length ? "Start" : "Nothing to answer yet",
      disabled: !CASES.length,
      onclick: startSurvey,
    });
    children.push(el("div", { class: "row" }, [startBtn]));

    app.appendChild(el("div", { class: "panel" }, children));
  }

  function renderCase() {
    var c = CASES[state.index];
    if (!c) {
      state.screen = "done";
      return render();
    }
    state.caseShownAt = Date.now();
    var resp = state.responses[c.id] || {};
    var orderIds = presentedOrder(c);
    var byId = {};
    (c.options || []).forEach(function (o) {
      byId[o.id] = o;
    });

    var wrap = el("div", {}, []);
    if (c.operation) wrap.appendChild(el("div", { class: "chip", text: c.operation }));
    wrap.appendChild(el("div", { class: "prompt", text: c.prompt || "" }));
    if (c.context) wrap.appendChild(el("p", { class: "context", text: c.context }));

    var grid = el("div", { class: "options" }, []);
    orderIds.forEach(function (oid, i) {
      var o = byId[oid];
      if (!o) return;
      var isSel = resp.choiceId === o.id;
      var card = el("div", {
        class: "option" + (isSel ? " selected" : ""),
        role: "button",
        tabindex: "0",
        "data-oid": o.id,
        onclick: function () {
          selectOption(c.id, o.id);
        },
      }, []);
      card.appendChild(el("div", { class: "option-key", text: String(i + 1) }));
      card.appendChild(el("div", { class: "option-label", text: o.label || o.id }));
      if (o.description) card.appendChild(el("div", { class: "option-desc", text: o.description }));

      if (o.previewText != null) {
        var pre = el("pre", { text: String(o.previewText) });
        card.appendChild(el("div", { class: "preview" }, [pre]));
      } else if (o.image) {
        var img = el("img", { src: o.image, alt: (o.label || o.id) + " preview" });
        card.appendChild(el("div", { class: "preview" }, [img]));
      } else if (o.preview != null) {
        card.appendChild(el("div", { class: "preview", html: String(o.preview) }));
      }
      grid.appendChild(card);
    });
    wrap.appendChild(grid);

    if (CONFIG.allowComments) {
      var ta = el("textarea", {
        placeholder: "Optional: why? (anything that made you pick this)",
        value: resp.comment || "",
        oninput: function (e) {
          var r = state.responses[c.id] || (state.responses[c.id] = {});
          r.comment = e.target.value;
          save();
        },
      });
      wrap.appendChild(el("div", { class: "field comment" }, [ta]));
    }

    var canAdvance = !CONFIG.requireChoice || !!resp.choiceId;
    var isLast = state.index === CASES.length - 1;
    var nav = el("div", { class: "navbar" }, [
      el("button", { class: "btn btn-ghost", text: "Back", onclick: goBack }),
      el("span", { class: "spacer" }, []),
      !canAdvance ? el("span", { class: "hint", text: "Pick an option to continue" }) : null,
      el("button", {
        class: "btn btn-primary",
        text: isLast ? "Finish" : "Next",
        disabled: !canAdvance,
        onclick: goNext,
      }),
    ]);
    wrap.appendChild(nav);
    app.appendChild(wrap);
  }

  function renderDone() {
    state.finishedAt = state.finishedAt || Date.now();
    save();

    var list = el("ul", { class: "summary-list" }, []);
    CASES.forEach(function (c) {
      var r = state.responses[c.id] || {};
      var opt = (c.options || []).filter(function (o) {
        return o.id === r.choiceId;
      })[0];
      var pick = opt
        ? el("span", { class: "pick", text: opt.label || opt.id })
        : el("span", { class: "pick none", text: "— not answered" });
      list.appendChild(
        el("li", {}, [el("span", { class: "op", text: c.operation || c.prompt || c.id }), pick])
      );
    });

    var children = [
      el("h1", { text: "All done — thank you!" }),
      el("p", {
        text:
          "You answered " + answeredCount() + " of " + CASES.length +
          " cases. Download your results below" + (CONFIG.returnTo ? ", then:" : "."),
      }),
    ];
    if (CONFIG.returnTo) children.push(el("p", { text: CONFIG.returnTo }));
    children.push(list);
    children.push(
      el("div", { class: "row" }, [
        el("button", { class: "btn btn-primary", text: "Download my results", onclick: downloadResults }),
        el("button", { class: "btn btn-ghost", text: "Back to last case", onclick: function () {
          state.screen = "case";
          state.index = Math.max(0, CASES.length - 1);
          render();
        } }),
        el("button", { class: "btn btn-ghost", text: "Start over", onclick: startOver }),
      ])
    );
    app.appendChild(el("div", { class: "panel" }, children));
  }

  // ------------------------------------------------------------- transitions
  function startSurvey() {
    if (!CASES.length) return;
    if (!state.startedAt) state.startedAt = Date.now();
    state.screen = "case";
    state.index = 0;
    save();
    render();
  }

  function selectOption(caseId, optionId) {
    var r = state.responses[caseId] || (state.responses[caseId] = {});
    r.choiceId = optionId;
    save();
    render(); // re-render to reflect selection + enable Next
  }

  function goNext() {
    var c = CASES[state.index];
    var resp = state.responses[c.id] || {};
    if (CONFIG.requireChoice && !resp.choiceId) return;
    stampDwell();
    if (state.index >= CASES.length - 1) {
      state.screen = "done";
      state.finishedAt = Date.now();
    } else {
      state.index += 1;
    }
    save();
    render();
  }

  function goBack() {
    stampDwell();
    if (state.index <= 0) {
      state.screen = "intro";
    } else {
      state.index -= 1;
    }
    save();
    render();
  }

  function startOver() {
    if (!window.confirm("Clear your answers and start over?")) return;
    clearSaved();
    state.index = 0;
    state.responses = {};
    state.order = {};
    state.startedAt = null;
    state.finishedAt = null;
    state.screen = "intro";
    render();
  }

  // --------------------------------------------------------------- results IO
  function buildResults() {
    return {
      surveyId: CONFIG.surveyId,
      surveyTitle: CONFIG.title,
      schemaVersion: CONFIG.schemaVersion,
      generatedAt: new Date().toISOString(),
      startedAt: state.startedAt ? new Date(state.startedAt).toISOString() : null,
      finishedAt: state.finishedAt ? new Date(state.finishedAt).toISOString() : null,
      durationMs: state.startedAt && state.finishedAt ? state.finishedAt - state.startedAt : null,
      userAgent: navigator.userAgent,
      participant: state.participant,
      totalCases: CASES.length,
      answeredCases: answeredCount(),
      responses: CASES.map(function (c) {
        var r = state.responses[c.id] || {};
        var opt = (c.options || []).filter(function (o) {
          return o.id === r.choiceId;
        })[0];
        return {
          caseId: c.id,
          operation: c.operation || "",
          prompt: c.prompt || "",
          presentedOrder: state.order[c.id] || (c.options || []).map(function (o) { return o.id; }),
          choiceId: r.choiceId || null,
          choiceLabel: opt ? opt.label || opt.id : null,
          comment: r.comment || "",
          dwellMs: r.dwellMs || 0,
        };
      }),
    };
  }

  function downloadResults() {
    var data = buildResults();
    var stamp = new Date().toISOString().replace(/[:.]/g, "-");
    var name = "ux-survey_" + CONFIG.surveyId + "_" + slug(state.participant.name) + "_" + stamp + ".json";
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = el("a", { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  // --------------------------------------------------------------- keyboard
  function onKey(e) {
    var t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    if (state.screen !== "case") return;
    var c = CASES[state.index];
    if (!c) return;
    if (e.key >= "1" && e.key <= "9") {
      var idx = parseInt(e.key, 10) - 1;
      var orderIds = presentedOrder(c);
      if (idx < orderIds.length) {
        selectOption(c.id, orderIds[idx]);
        e.preventDefault();
      }
    } else if (e.key === "Enter") {
      goNext();
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      goBack();
      e.preventDefault();
    }
  }

  // --------------------------------------------------------------- boot
  loadSaved();
  if (CONFIG.returnTo) footerNote.textContent = CONFIG.returnTo;
  document.addEventListener("keydown", onKey);
  render();
})();
