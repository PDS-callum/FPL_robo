
    // Keep reload on Pulse — strip leftover section hashes from older nav.
    try {
      if ("scrollRestoration" in history) history.scrollRestoration = "manual";
      if (location.hash) {
        history.replaceState(null, "", location.pathname + location.search);
      }
    } catch (_) { /* ignore */ }

    const $ = (id) => document.getElementById(id);

    function esc(s) {
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function fmtHours(h) {
      if (h == null || Number.isNaN(h)) return "—";
      if (h >= 48) return `${(h / 24).toFixed(1)}d`;
      if (h >= 1) return `${h.toFixed(1)}h`;
      return `${Math.max(0, Math.round(h * 60))}m`;
    }

    function fmtMoney(v) {
      if (v == null || Number.isNaN(v)) return "—";
      return `£${Number(v).toFixed(1)}m`;
    }

    function authClass(ok, seconds) {
      if (!ok) return "bad";
      if (seconds != null && seconds < 3600) return "warn";
      return "";
    }

    function metric(label, value, hint) {
      return `<div class="metric"><div class="label">${label}</div><div class="value" data-metric-value>${value}</div>${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
    }

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const liteFx = !!(window.__FPL_LITE_FX__ || reduceMotion || document.documentElement.classList.contains("lite-fx"));
    if (liteFx) document.documentElement.classList.add("lite-fx");

    const VIEW_KEY = "fpl_robo_view";
    let statusExpanded = false;
    let logsOpen = false;

    function polishMotion() {
      // Bars/ring are authored at 0% so desktop can animate in. On lite/mobile
      // we skip animation — but must still snap to the final widths/offsets.
      if (liteFx) {
        document.querySelectorAll(".bar-fill[data-w], .hbar-fill[data-w]").forEach((el) => {
          el.style.width = el.getAttribute("data-w") || "0%";
        });
        document.querySelectorAll(".ring-progress[data-target-offset]").forEach((el) => {
          el.style.strokeDashoffset = el.getAttribute("data-target-offset");
        });
        return;
      }
      // Animate chip / deadline bars from 0 after paint.
      document.querySelectorAll(".bar-fill[data-w], .hbar-fill[data-w]").forEach((el) => {
        const w = el.getAttribute("data-w");
        el.style.width = "0%";
        requestAnimationFrame(() => {
          requestAnimationFrame(() => { el.style.width = w; });
        });
      });
      // Deadline ring ease-in.
      document.querySelectorAll(".ring-progress[data-target-offset]").forEach((el) => {
        const target = el.getAttribute("data-target-offset");
        el.style.strokeDashoffset = el.getAttribute("stroke-dasharray") || el.style.strokeDasharray;
        requestAnimationFrame(() => {
          requestAnimationFrame(() => { el.style.strokeDashoffset = target; });
        });
      });
    }

    function initSectionReveal() {
      // App-shell: active panel is always visible; no scroll-into-view reveal.
    }

    function setSectionTitle(id, num, label) {
      const el = $(id);
      if (!el) return;
      if (num == null || num === "") {
        el.textContent = label;
        return;
      }
      el.innerHTML = `<span class="sec-num">${num}</span>${esc(label)}`;
    }

    function syncMovesPaneHeight() {
      const moves = document.querySelector(".squad-pane--moves");
      const section = document.getElementById("squad");
      if (!moves || !section) return;
      if (window.matchMedia("(max-width: 860px)").matches) {
        moves.style.height = "";
        moves.style.maxHeight = "";
        return;
      }
      // Cap to the Squad scrollport so #movesList scrolls inside the box.
      const head = section.querySelector(":scope > h2");
      const sub = section.querySelector(":scope > p.sub");
      const tabs = section.querySelector(".squad-tabs");
      const tabsVisible = tabs && getComputedStyle(tabs).display !== "none";
      const reserved =
        (head ? head.offsetHeight : 0) +
        (sub ? sub.offsetHeight : 0) +
        (tabsVisible ? tabs.offsetHeight + 8 : 0) +
        28;
      const h = Math.max(220, Math.floor(section.clientHeight - reserved));
      moves.style.height = `${h}px`;
      moves.style.maxHeight = `${h}px`;
    }

    function initSquadTabs() {
      const tabs = Array.from(document.querySelectorAll("[data-squad-tab]"));
      const panes = Array.from(document.querySelectorAll("[data-squad-pane]"));
      if (!tabs.length || !panes.length) return;

      function showTab(id) {
        tabs.forEach((btn) => {
          const on = btn.getAttribute("data-squad-tab") === id;
          btn.classList.toggle("is-active", on);
          btn.setAttribute("aria-selected", on ? "true" : "false");
        });
        panes.forEach((pane) => {
          pane.classList.toggle("is-active", pane.getAttribute("data-squad-pane") === id);
        });
        requestAnimationFrame(syncMovesPaneHeight);
      }

      tabs.forEach((btn) => {
        btn.addEventListener("click", () => showTab(btn.getAttribute("data-squad-tab")));
      });
      showTab("xi");
      window.addEventListener("resize", () => {
        clearTimeout(syncMovesPaneHeight._t);
        syncMovesPaneHeight._t = setTimeout(syncMovesPaneHeight, 80);
      });
    }

    function fmtRank(n) {
      if (n == null || Number.isNaN(Number(n))) return "—";
      return Number(n).toLocaleString();
    }

    function fmtPts(n) {
      if (n == null || Number.isNaN(Number(n))) return "—";
      return String(Math.round(Number(n)));
    }

    function niceMax(v) {
      if (!Number.isFinite(v) || v <= 0) return 1;
      const p = Math.pow(10, Math.floor(Math.log10(v)));
      return Math.ceil(v / p) * p;
    }

    function renderGwChart(rows) {
      if (!rows.length) {
        return `<div class="empty">No finished gameweeks yet — points appear once GW1 scores.</div>`;
      }
      const W = 420, H = 180, padL = 28, padR = 10, padT = 12, padB = 28;
      const maxPts = niceMax(Math.max(...rows.map((r) => Number(r.points) || 0), 1));
      const maxCum = niceMax(Math.max(...rows.map((r) => Number(r.total_points) || 0), 1));
      const n = rows.length;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const gap = 0.28;
      const bw = plotW / Math.max(n, 1);
      const bars = rows.map((r, i) => {
        const pts = Number(r.points) || 0;
        const h = (pts / maxPts) * plotH;
        const x = padL + i * bw + bw * gap / 2;
        const y = padT + plotH - h;
        const hit = Number(r.transfers_cost) > 0;
        return `<rect class="bar ${hit ? "hit" : ""}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw * (1 - gap)).toFixed(1)}" height="${Math.max(1, h).toFixed(1)}"><title>GW${r.gameweek}: ${pts} pts</title></rect>`;
      }).join("");
      const cumPts = rows.map((r, i) => {
        const x = padL + i * bw + bw / 2;
        const y = padT + plotH - ((Number(r.total_points) || 0) / maxCum) * plotH;
        return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      const labels = rows.filter((_, i) => n <= 12 || i % Math.ceil(n / 10) === 0 || i === n - 1).map((r, _, arr) => {
        const i = rows.indexOf(r);
        const x = padL + i * bw + bw / 2;
        return `<text class="axis" x="${x.toFixed(1)}" y="${H - 8}" text-anchor="middle">GW${r.gameweek}</text>`;
      }).join("");
      return `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Gameweek points">
        <line class="grid" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" />
        <line class="grid" x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" />
        <text class="axis" x="${padL - 4}" y="${padT + 4}" text-anchor="end">${maxPts}</text>
        <text class="axis" x="${padL - 4}" y="${padT + plotH}" text-anchor="end">0</text>
        ${bars}
        <path class="cum" d="${cumPts}" />
        ${labels}
      </svg>
      <div class="chart-legend"><span class="gw">GW points</span><span class="cum">Cumulative</span></div>`;
    }

    function renderEvChart(snaps) {
      const usable = (snaps || []).filter((s) => s.plan_objective != null || s.xi_ev != null);
      if (usable.length < 2) {
        return `<div class="empty">Collecting snapshots… a trend line appears after a few daemon / dashboard cycles.</div>`;
      }
      const W = 420, H = 180, padL = 32, padR = 10, padT = 12, padB = 28;
      const vals = usable.flatMap((s) => [Number(s.plan_objective), Number(s.xi_ev)].filter((v) => Number.isFinite(v)));
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const span = Math.max(0.5, hi - lo);
      const yMin = lo - span * 0.08;
      const yMax = hi + span * 0.08;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xAt = (i) => padL + (usable.length === 1 ? plotW / 2 : (i / (usable.length - 1)) * plotW);
      const yAt = (v) => padT + plotH - ((Number(v) - yMin) / (yMax - yMin)) * plotH;
      const line = (key) => {
        const pts = usable
          .map((s, i) => (s[key] == null || !Number.isFinite(Number(s[key])) ? null : [xAt(i), yAt(s[key])]))
          .filter(Boolean);
        if (!pts.length) return "";
        return pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
      };
      const evPath = line("plan_objective");
      const xiPath = line("xi_ev");
      const areaPts = usable.map((s, i) => {
        const v = s.plan_objective;
        if (v == null) return null;
        return `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`;
      }).filter(Boolean);
      const area = areaPts.length
        ? `M${areaPts[0]} L${areaPts.slice(1).join(" L")} L${areaPts[areaPts.length - 1].split(",")[0]},${(padT + plotH).toFixed(1)} L${areaPts[0].split(",")[0]},${(padT + plotH).toFixed(1)} Z`
        : "";
      const first = usable[0].ts ? new Date(usable[0].ts) : null;
      const last = usable[usable.length - 1].ts ? new Date(usable[usable.length - 1].ts) : null;
      const fmtD = (d) => d && !Number.isNaN(d.getTime())
        ? d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
        : "";
      return `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Bot EV over time">
        <line class="grid" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" />
        <line class="grid" x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" />
        <text class="axis" x="${padL - 4}" y="${padT + 4}" text-anchor="end">${yMax.toFixed(0)}</text>
        <text class="axis" x="${padL - 4}" y="${padT + plotH}" text-anchor="end">${yMin.toFixed(0)}</text>
        ${area ? `<path class="area-ev" d="${area}" />` : ""}
        ${evPath ? `<path class="line-ev" d="${evPath}" />` : ""}
        ${xiPath ? `<path class="line-xi" d="${xiPath}" />` : ""}
        <text class="axis" x="${padL}" y="${H - 8}">${esc(fmtD(first))}</text>
        <text class="axis" x="${W - padR}" y="${H - 8}" text-anchor="end">${esc(fmtD(last))}</text>
      </svg>
      <div class="chart-legend"><span>Plan EV</span><span class="xi">XI EV</span></div>`;
    }

    function deadlineRing(stats) {
      const hours = stats.deadline_hours;
      const fill = stats.deadline_fill == null ? 0 : Number(stats.deadline_fill);
      const r = 44;
      const c = 2 * Math.PI * r;
      const offset = c * (1 - Math.min(1, Math.max(0, fill)));
      const label = hours == null ? "—" : fmtHours(hours);
      const CHIP_CAPS = { TC: 6, BB: 8, FH: 12, WC: 18, triple_captain: 6, bench_boost: 8, free_hit: 12, wildcard: 18 };
      const chipBars = (stats.chip_bars || []).map((b) => {
        const score = Number(b.score) || 0;
        // Prefer API cap; for legacy payloads, never treat 2× TC score against a captain-EV cap.
        let cap = Number(b.cap);
        if (!(cap > 0)) cap = CHIP_CAPS[b.label] || CHIP_CAPS[b.key] || 1;
        let fill = (b.fill != null && Number(b.cap) > 0)
          ? Math.min(1, Math.max(0, Number(b.fill)))
          : Math.min(1, Math.max(0, score / cap));
        // Keep a visible stub whenever there is any score, so thin fills still read.
        const pct = fill <= 0 ? 0 : Math.max(6, Math.round(fill * 100));
        const play = b.play ? " play" : "";
        const unit = b.label === "TC" ? "captain EV" : "score";
        const title = `${b.label}: ${score.toFixed(1)} ${unit} of ${cap.toFixed(1)} play threshold (${Math.round(fill * 100)}%)`;
        return `<div class="bar-row${play}" title="${esc(title)}">
          <span>${esc(b.label)}${b.play ? " ·" : ""}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${pct}%" style="width:0%"></div></div>
          <span class="score">${score.toFixed(1)}<small>/${cap.toFixed(1)}</small></span>
        </div>`;
      }).join("");
      return `
        <h3>Deadline & chips</h3>
        <div class="ring-row">
          <svg class="ring" viewBox="0 0 108 108" aria-hidden="true">
            <circle cx="54" cy="54" r="${r}" stroke="rgba(255,255,255,0.12)" stroke-width="8" fill="none" />
            <circle class="ring-progress" cx="54" cy="54" r="${r}" stroke="#2dd4bf" stroke-width="8" fill="none"
              stroke-linecap="butt" stroke-dasharray="${c}" stroke-dashoffset="${c}"
              data-target-offset="${offset}"
              transform="rotate(-90 54 54)" />
            <text x="54" y="52">${label}</text>
            <text class="label" x="54" y="68">TO DEADLINE</text>
          </svg>
          <div>
            <div class="muted">Chip scores vs play threshold</div>
            <div class="bars" style="margin-top:0.55rem">${chipBars || `<div class="muted">No chip scores yet.</div>`}</div>
            ${(() => {
              const sched = (stats.chip_schedule && stats.chip_schedule.applied) || [];
              const gate = (stats.chip_lineup_gate && stats.chip_lineup_gate.suppressed) || [];
              const notes = [...sched, ...gate.map((g) => `gate:${g}`)];
              if (!notes.length) return "";
              return `<div class="chip-notes">${notes.map((n) => `<div>${esc(n)}</div>`).join("")}</div>`;
            })()}
          </div>
        </div>`;
    }

    function playerPhotoUrl(code, size = "110x140") {
      if (code == null || code === "") return null;
      return `https://resources.premierleague.com/premierleague/photos/players/${size}/p${code}.png`;
    }

    function playerShirtUrl(p) {
      const code = p && p.team_code;
      if (code == null || code === "") return null;
      const gk = String((p && p.position) || "").toUpperCase() === "GK";
      return gk
        ? `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${code}_1-66.png`
        : `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${code}-66.png`;
    }

    function playerPhotoHtml(p, { className = "photo", size = "110x140" } = {}) {
      // Prefer current-club FPL kits: PL headshots often lag transfers (e.g. João Pedro still in BHA).
      const kit = playerShirtUrl(p);
      const head = playerPhotoUrl(p && p.photo_code, size);
      const initial = esc(String((p && (p.web_name || p.first_name)) || "?").slice(0, 1).toUpperCase());
      if (!kit && !head) {
        return `<span class="${className} is-fallback" aria-hidden="true">${initial}</span>`;
      }
      const src = kit || head;
      const kitClass = kit ? " is-kit" : "";
      const headAttr = (!kit && head) ? "" : (head ? ` data-head="${esc(head)}"` : "");
      return `<img class="${className}${kitClass}" src="${esc(src)}" alt="" loading="lazy" decoding="async" referrerpolicy="no-referrer"${headAttr} data-fallback="${initial}" onerror="(function(img){if(img.dataset.head&&img.src!==img.dataset.head){img.src=img.dataset.head;img.classList.remove('is-kit');return;}img.outerHTML='<span class=\\'${className} is-fallback\\' aria-hidden=\\'true\\'>'+img.dataset.fallback+'</span>';})(this)" />`;
    }

    function statusLabel(code) {
      const map = {
        a: "Available",
        d: "Doubtful",
        i: "Injured",
        s: "Suspended",
        u: "Unavailable",
        n: "Unavailable",
      };
      return map[String(code || "").toLowerCase()] || (code ? String(code) : "—");
    }

    function closePlayerSheet() {
      const sheet = $("playerSheet");
      if (!sheet) return;
      sheet.classList.remove("is-open");
      sheet.hidden = true;
    }

    function openPlayerSheet(player) {
      const sheet = $("playerSheet");
      if (!sheet || !player) return;
      const full = [player.first_name, player.second_name || player.web_name].filter(Boolean).join(" ")
        || player.web_name
        || `Player ${player.id}`;
      $("playerSheetName").textContent = full;
      const role = [
        player.position,
        player.team,
        player.captain ? "Captain" : (player.vice_captain ? "Vice" : null),
      ].filter(Boolean).join(" · ");
      const chance = player.chance_of_playing_next_round;
      $("playerSheetMeta").textContent = [
        role,
        statusLabel(player.status),
        chance != null ? `${chance}% next` : null,
      ].filter(Boolean).join(" · ");

      const photo = $("playerSheetPhoto");
      const kit = playerShirtUrl(player);
      const head = playerPhotoUrl(player.photo_code, "250x250");
      if (photo) {
        photo.classList.remove("is-kit");
        photo.onerror = null;
        if (kit || head) {
          photo.hidden = false;
          if (kit) {
            photo.classList.add("is-kit");
            photo.src = kit;
            photo.onerror = () => {
              if (head) {
                photo.onerror = () => { photo.hidden = true; };
                photo.classList.remove("is-kit");
                photo.src = head;
              } else {
                photo.hidden = true;
              }
            };
          } else {
            photo.src = head;
            photo.onerror = () => { photo.hidden = true; };
          }
        } else {
          photo.hidden = true;
          photo.removeAttribute("src");
        }
      }

      const stat = (k, v) => {
        if (v == null || v === "" || Number.isNaN(v)) return "";
        return `<div class="player-sheet-stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;
      };
      const num = (v, digits = 1) => {
        if (v == null || v === "") return null;
        const n = Number(v);
        return Number.isFinite(n) ? n.toFixed(digits) : String(v);
      };
      $("playerSheetStats").innerHTML = [
        stat("Vector EV", num(player.ev, 2)),
        stat("EP anchor", num(player.ep_anchor != null ? player.ep_anchor : player.ep_next, 2)),
        stat("Residual", num(player.residual_scaled != null ? player.residual_scaled : player.residual, 2)),
        stat("Price", player.price != null ? fmtMoney(player.price) : null),
        stat("EP next", num(player.ep_next, 1)),
        stat("Form", num(player.form, 1)),
        stat("Selected", player.selected_by_percent != null ? `${player.selected_by_percent}%` : null),
        stat("PPG", num(player.points_per_game, 1)),
        (() => {
          const mins = Number(player.minutes);
          const starts = Number(player.starts);
          if (Number.isFinite(mins) && Number.isFinite(starts) && starts > 0) {
            return stat("Avg mins", String(Math.min(90, Math.round(mins / starts))));
          }
          if (Number.isFinite(mins) && mins > 0) return stat("Minutes", String(mins));
          return "";
        })(),
        stat("G / A", (player.goals_scored != null || player.assists != null)
          ? `${player.goals_scored ?? 0} / ${player.assists ?? 0}` : null),
        stat("CS", player.clean_sheets != null ? String(player.clean_sheets) : null),
        stat("Bonus", player.bonus != null ? String(player.bonus) : null),
        stat("xGI/90", num(player.xgi90, 2)),
        stat("xG / xA", (player.expected_goals != null || player.expected_assists != null)
          ? `${num(player.expected_goals, 2) ?? "—"} / ${num(player.expected_assists, 2) ?? "—"}` : null),
        stat("Next FDR", player.next_fdr != null ? String(player.next_fdr) : null),
        stat("ICT", num(player.ict_index, 1)),
        stat("Set piece", num(player.set_piece_score, 2)),
        stat("Quality", num(player.quality, 2)),
      ].join("");

      const note = $("playerSheetNote");
      if (note) {
        const news = String(player.news || "").trim();
        if (news) {
          note.hidden = false;
          note.textContent = news;
        } else {
          note.hidden = true;
          note.textContent = "";
        }
      }

      sheet.hidden = false;
      requestAnimationFrame(() => sheet.classList.add("is-open"));
    }

    function bindPlayerClicks(root, byId) {
      if (!root) return;
      root.querySelectorAll("[data-player-id]").forEach((el) => {
        el.addEventListener("click", () => {
          const id = Number(el.getAttribute("data-player-id"));
          const player = byId.get(id);
          if (player) openPlayerSheet(player);
        });
      });
    }

    function renderPitch(xi, captainId, viceId) {
      if (!xi.length) return `<div class="empty" style="border:0;color:rgba(255,255,255,0.7)">No XI yet.</div>`;
      const rows = { GK: [], DEF: [], MID: [], FWD: [] };
      xi.forEach((p) => {
        const pos = String(p.position || "");
        if (rows[pos]) rows[pos].push(p);
      });
      // Sort each line by EV so the eye reads strongest players first left→right.
      Object.keys(rows).forEach((pos) => {
        rows[pos].sort((a, b) => (Number(b.ev) || 0) - (Number(a.ev) || 0));
      });
      let delay = 0;
      const shirt = (p) => {
        const cap = p.captain || (captainId != null && p.id === captainId);
        const vice = !cap && (p.vice_captain || (viceId != null && p.id === viceId));
        const d = (delay++ * 0.04).toFixed(2);
        const cls = ["shirt", cap ? "captain" : "", vice ? "vice" : ""].filter(Boolean).join(" ");
        const badge = cap ? "C" : (vice ? "VC" : "");
        const label = [p.web_name || p.id, p.team, p.position].filter(Boolean).join(", ");
        return `<button type="button" class="${cls}" style="animation-delay:${d}s" data-player-id="${esc(p.id)}" aria-label="${esc(label)}">
          <span class="badge" ${badge ? "" : "hidden"}>${badge}</span>
          ${playerPhotoHtml(p)}
          <div class="nm">${esc(p.web_name || p.id)}</div>
          <div class="ev">${p.ev != null ? Number(p.ev).toFixed(1) : "—"}</div>
          ${p.residual_scaled != null && Number(p.residual_scaled) !== 0
            ? `<div class="pr" style="opacity:0.85">${Number(p.residual_scaled) > 0 ? "+" : ""}${Number(p.residual_scaled).toFixed(1)}</div>`
            : `<div class="pr">${fmtMoney(p.price)}</div>`}
          ${p.team ? `<span class="team">${esc(p.team)}</span>` : ""}
        </button>`;
      };
      // FPL-style: attack at top, keeper at bottom.
      const lines = ["FWD", "MID", "DEF", "GK"]
        .map((pos) => {
          const list = rows[pos] || [];
          if (!list.length) return "";
          return `<div class="pitch-row" style="--n:${list.length}" data-n="${list.length}">${list.map(shirt).join("")}</div>`;
        })
        .join("");
      return `<div class="pitch-mark top" aria-hidden="true"></div>
        <div class="pitch-mark bottom" aria-hidden="true"></div>
        <div class="pitch-inner">${lines}</div>`;
    }

    function fmtWhen(iso) {
      if (!iso) return "never";
      try {
        return new Date(iso).toLocaleString();
      } catch (_) {
        return String(iso);
      }
    }

    function renderSchedule(sched, deadline) {
      const cfg = sched.config || {};
      const hours = sched.hours_to_deadline;
      const phaseLabel = sched.phase_label || sched.phase || "—";
      const daily = cfg.daily_ingest || "08:30";

      $("scheduleSub").textContent = hours != null
        ? `${phaseLabel} · ${fmtHours(hours)} to GW${deadline.gameweek ?? "—"} lock`
        : "Daemon cadence for news and planning.";

      const nextNews = (sched.next || []).find((n) => n.title === "Daily news");
      const preDl = cfg.pre_deadline_hours ?? 0.5;
      $("scheduleMetrics").innerHTML = [
        metric("Phase", phaseLabel, sched.run_lock_held ? "Run in progress" : `Checks every ${Math.round((sched.current_poll_seconds || 0) / 60)}m`),
        metric("Daily news", daily, nextNews && nextNews.in_hours != null ? `next in ${fmtHours(nextNews.in_hours)}` : (cfg.timezone || "").split("/").pop()),
        metric("Final scrape", fmtHours(preDl), "Before FPL deadline · skips daily if deadline is earlier that morning"),
        metric("Writes", fmtHours(cfg.execute_within_hours), sched.unlimited_transfers ? "Unlimited now" : (sched.can_write_now ? "Window open" : "Dry-run until then")),
      ].join("");

      const nearH = Number(cfg.near_deadline_hours || 1);
      const execH = Number(cfg.execute_within_hours || 0.5);
      const scrapeH = Number(preDl);
      const span = Math.max(6, nearH * 3);
      const markers = [
        { hours: scrapeH, label: "Scrape" },
        { hours: execH, label: "Write" },
        { hours: 0, label: "Lock" },
      ];
      const pct = (h) => {
        const v = Math.max(0, Math.min(span, Number(h)));
        return ((span - v) / span) * 100;
      };
      const nowH = hours == null ? null : Math.max(0, Number(hours));
      const nowPct = nowH == null ? null : (nowH > span ? 0 : pct(nowH));

      // Merge markers that land on the same spot (e.g. scrape + write both at 0.5h).
      const merged = [];
      markers.forEach((mk) => {
        const p = Number(pct(mk.hours).toFixed(1));
        const prev = merged.find((m) => Math.abs(m.pct - p) < 0.6);
        if (prev) prev.labels.push(mk.label);
        else merged.push({ pct: p, hours: mk.hours, labels: [mk.label] });
      });
      const markerHtml = merged.map((mk, i) => {
        const active = mk.labels.some((label) =>
          (sched.phase === "near_deadline" && label === "News")
          || (sched.phase === "execute" && label === "Write")
          || (sched.phase === "locked" && label === "Lock")
        );
        const isEnd = mk.pct >= 98;
        const isStart = mk.pct <= 2;
        const nearEnd = !isEnd && mk.pct >= 82;
        // Keep Lock below; put crowded near-end labels above and inward.
        const stack = isEnd || i % 2 === 1 ? "below" : "above";
        const edge = isStart ? " is-start" : (isEnd ? " is-end" : (nearEnd ? " is-near-end" : ""));
        return `<div class="sched-marker sched-marker--${stack}${edge}${active ? " is-active" : ""}" style="left:${mk.pct.toFixed(1)}%">
          <span class="lbl">${esc(mk.labels.join(" · "))}</span>
        </div>`;
      }).join("");

      $("scheduleTrack").innerHTML = `
        <div class="sched-track-head">
          <strong>To deadline</strong>
          <span>${hours != null ? `${esc(fmtHours(hours))} left` : "—"}</span>
        </div>
        <div class="sched-rail" aria-hidden="true">
          <div class="sched-rail-fill" style="width:${(nowPct == null ? 0 : nowPct).toFixed(1)}%"></div>
          ${nowPct != null ? `<div class="sched-now${nowPct <= 2 ? " is-start" : (nowPct >= 98 ? " is-end" : "")}" style="left:${nowPct.toFixed(1)}%">Now</div>` : ""}
          ${markerHtml}
        </div>`;

      $("scheduleJobs").innerHTML = (sched.jobs || []).map((job) => {
        const on = !!job.active;
        const extra = job.last_kind
          ? ` · ${job.last_kind}`
          : ((job.last_triggers || []).length ? ` · ${(job.last_triggers || []).slice(0, 2).join(", ")}` : "");
        return `<article class="sched-job ${on ? "is-on" : "is-off"}">
          <div class="job-top">
            <h3>${esc(job.title)}</h3>
            <span class="state">${on ? "Due soon" : "Idle"}</span>
          </div>
          <p class="detail">${esc(job.when || "")}</p>
          <div class="meta">Last ${esc(fmtWhen(job.last_at))}${esc(extra)}</div>
        </article>`;
      }).join("") || `<div class="empty">Schedule unavailable.</div>`;
    }

    function render(data) {
      const m = data.manager || {};
      const d = data.deadline || {};
      const s = data.status || {};
      const plan = data.plan || {};
      const stats = data.stats || {};

      $("lede").textContent =
        `${m.team_name || "Your team"} · ${m.name || "manager"} · planning GW${d.gameweek ?? "—"}` +
        (stats.formation ? ` · ${stats.formation}` : "");

      const phaseLabel = s.unlimited_transfers
        ? "Unlimited"
        : s.in_execute_window
          ? "Execute open"
          : s.near_deadline
            ? "Near deadline"
            : "Watching";
      const writeLabel = s.can_write_now ? "Can write" : "Dry-run";
      const authLabel = s.auth_ok ? "Auth OK" : "Auth issue";

      const residual = data.meta && data.meta.residual_fit;
      const residualPill = residual
        ? `<span class="pill">Ridge ${residual.fitted ? `n=${residual.n_train ?? "—"}` : "hand"} · ${esc(String(residual.source || "").split(":")[0] || "—")}</span>`
        : "";
      const gatePill = (data.meta && data.meta.gate_reject)
        ? `<span class="pill">Gate ${esc(data.meta.gate_reject)}</span>`
        : "";
      const consensusPill = (data.meta && data.meta.consensus_mode)
        ? `<span class="pill">SA ${esc(data.meta.consensus_mode)}</span>`
        : "";
      const shadow = data.meta && data.meta.shadow_mip;
      const shadowPill = shadow && shadow.objective != null && plan.objective != null
        ? `<span class="pill">MIP Δ ${(Number(shadow.objective) - Number(plan.objective)).toFixed(1)}</span>`
        : "";

      $("statusPills").innerHTML = [
        `<span class="pill"><span class="dot ${authClass(s.auth_ok, s.auth_seconds_remaining)}"></span>${s.auth_ok ? "Authenticated" : "Auth issue"}</span>`,
        `<span class="pill">EV ${(data.meta && data.meta.ev_mode) || "—"}</span>`,
        `<span class="pill">${esc((data.meta && data.meta.optimiser) || (plan.optimiser) || "—")}</span>`,
        `<span class="pill">${esc((data.meta && data.meta.feature_version) || "—")}</span>`,
        residualPill,
        consensusPill,
        gatePill,
        shadowPill,
        `<span class="pill">Deadline ${fmtHours(d.hours_to_deadline)}</span>`,
        `<span class="pill">${s.unlimited_transfers ? "Unlimited transfers" : s.in_execute_window ? "Execute window OPEN" : s.near_deadline ? "Near deadline" : "Watching"}</span>`,
        `<span class="pill">${s.can_write_now ? "Can write" : "Dry-run only"}</span>`,
        `<span class="pill">${plan.transfers_made ?? 0} transfers · ${plan.hits ?? 0} hits</span>`,
        `<span class="pill">Signals ${(data.intel && data.intel.summary && data.intel.summary.total != null) ? data.intel.summary.total : (stats.signals && stats.signals.count) || 0}</span>`,
        `<span class="pill">${fmtPts((data.scoreboard && data.scoreboard.overall_points) ?? m.overall_points)} pts</span>`,
      ].filter(Boolean).join("");

      const compact = $("statusCompact");
      if (compact) {
        compact.innerHTML = [
          `<span class="dot ${authClass(s.auth_ok, s.auth_seconds_remaining)}"></span>`,
          `<span>${esc(authLabel)}</span>`,
          `<span class="sep">·</span>`,
          `<span>Deadline ${esc(fmtHours(d.hours_to_deadline))}</span>`,
          `<span class="sep">·</span>`,
          `<span>${esc(phaseLabel)}</span>`,
          `<span class="sep">·</span>`,
          `<span>${esc(writeLabel)}</span>`,
        ].join("");
      }
      const strip = $("statusStrip");
      const expandBtn = $("statusExpand");
      if (strip) strip.classList.toggle("is-expanded", statusExpanded);
      if (expandBtn) {
        expandBtn.setAttribute("aria-expanded", statusExpanded ? "true" : "false");
        expandBtn.textContent = statusExpanded ? "Less" : "More";
      }

      const meta = data.meta || {};
      const tunnelUrl = meta.public_tunnel_url;
      const tunnelAge = meta.tunnel_age_seconds;
      const authLeft = s.auth_seconds_remaining;
      const authDot = authClass(s.auth_ok, authLeft);
      let tunnelLabel = "No tunnel";
      if (tunnelUrl) {
        try {
          tunnelLabel = new URL(tunnelUrl).host.replace(".trycloudflare.com", "");
        } catch (_) {
          tunnelLabel = "Tunnel";
        }
        if (tunnelAge != null && tunnelAge > 86400) tunnelLabel += " · stale?";
      }
      const health = $("navHealth");
      if (health) {
        health.innerHTML = `
          <span class="dot ${authDot}" title="Auth"></span>
          <span>${s.auth_ok ? (authLeft != null ? fmtHours(authLeft / 3600) : "auth") : "auth"}</span>
          ${tunnelUrl ? `<a href="${esc(tunnelUrl)}" target="_blank" rel="noopener" title="${esc(tunnelUrl)}">${esc(tunnelLabel)}</a>` : `<span>${esc(tunnelLabel)}</span>`}
        `;
      }

      $("metrics").innerHTML = [
        metric("Plan EV", stats.plan_objective != null ? Number(stats.plan_objective).toFixed(2) : "—", "Optimiser objective"),
        metric("XI EV", stats.xi_ev != null ? Number(stats.xi_ev).toFixed(1) : "—", "Starting eleven"),
        metric("Bench EV", stats.bench_ev != null ? Number(stats.bench_ev).toFixed(1) : "—", "Cover quality"),
        metric("Squad value", fmtMoney(stats.squad_value), `XI ${fmtMoney(stats.xi_value)}`),
        metric("Bank", fmtMoney(m.bank), m.unlimited_transfers ? "Unlimited FTs" : `${m.free_transfers ?? plan.free_transfers ?? "—"} FT`),
        metric("Captain", stats.captain_name || plan.captain_name || "—", stats.captain_ev != null ? `EV ${Number(stats.captain_ev).toFixed(1)}` : ""),
        metric("Vice", stats.vice_captain_name || plan.vice_captain_name || "—", stats.vice_captain_ev != null ? `EV ${Number(stats.vice_captain_ev).toFixed(1)}` : ""),
      ].join("");

      renderSchedule(data.schedule || {}, d);

      const board = data.scoreboard || {};
      const hist = data.history || {};
      const pulseScore = $("pulseScore");
      if (pulseScore) {
        pulseScore.innerHTML = [
          metric("Overall", fmtPts(board.overall_points ?? m.overall_points), board.season_started ? "Season total" : "Awaiting GW1"),
          metric("This GW", fmtPts(board.event_points ?? m.event_points), board.event_rank != null ? `rank ${fmtRank(board.event_rank)}` : ""),
          metric("Rank", fmtRank(board.overall_rank ?? m.overall_rank), board.current_event != null ? `after GW${board.current_event}` : ""),
          `<button type="button" class="pulse-score-link" data-open-view="score">Score charts</button>`,
        ].join("");
        pulseScore.querySelectorAll("[data-open-view]").forEach((el) => {
          el.addEventListener("click", () => {
            if (typeof window.showDashboardView === "function") {
              window.showDashboardView(el.getAttribute("data-open-view"));
            }
          });
        });
      }
      $("scoreSub").textContent = board.season_started
        ? `Live FPL score for ${board.team_name || m.team_name || "your team"}.`
        : "Preseason — official points start after GW1. Bot EV snapshots still accumulate below.";
      $("scoreMetrics").innerHTML = [
        metric("Overall", fmtPts(board.overall_points ?? m.overall_points), board.season_started ? "Season total" : "Awaiting GW1"),
        metric("Rank", fmtRank(board.overall_rank ?? m.overall_rank), board.current_event != null ? `after GW${board.current_event}` : ""),
        metric("This GW", fmtPts(board.event_points ?? m.event_points), board.event_rank != null ? `GW rank ${fmtRank(board.event_rank)}` : ""),
        metric("GW scored", String((hist.gw_points || []).length), `${(hist.snapshots || []).length} bot snapshots`),
      ].join("");
      $("gwChartSub").textContent = (hist.gw_points || []).length
        ? `${(hist.gw_points || []).length} gameweeks · bars = GW points · line = cumulative`
        : "Official FPL points once gameweeks finish.";
      $("gwChart").innerHTML = renderGwChart(hist.gw_points || []);
      $("evChartSub").textContent = (hist.snapshots || []).length
        ? `${(hist.snapshots || []).length} snapshots (≈15 min cadence)`
        : "Plan objective and XI EV from dashboard snapshots.";
      $("evChart").innerHTML = renderEvChart(hist.snapshots || []);

      $("deadlinePanel").innerHTML = deadlineRing(stats);

      const xiBars = stats.xi_ev_bars || [];
      const maxEv = Math.max(0.01, ...xiBars.map((p) => Number(p.ev) || 0));
      $("evBars").innerHTML = xiBars.length
        ? xiBars.map((p) => {
            const pct = Math.round(((Number(p.ev) || 0) / maxEv) * 100);
            return `<div class="hbar-row ${p.captain ? "captain" : ""}">
              <span class="name">${p.web_name || "—"}</span>
              <div class="hbar-track"><div class="hbar-fill" data-w="${pct}%" style="width:0%"></div></div>
              <span>${Number(p.ev).toFixed(1)}</span>
            </div>`;
          }).join("")
        : `<div class="empty">No XI EV data yet.</div>`;

      const pos = stats.ev_by_position || [];
      const maxPos = Math.max(0.01, ...pos.map((p) => Number(p.ev) || 0));
      $("posChart").innerHTML = pos.map((p) => {
        const h = Math.max(4, Math.round(((Number(p.ev) || 0) / maxPos) * 120));
        return `<div class="pos-col"><div class="stem" style="height:0" data-h="${h}"></div><div class="val">${Number(p.ev).toFixed(1)}</div><div class="pos">${p.position} · ${p.n}</div></div>`;
      }).join("") || `<div class="empty">No position split yet.</div>`;
      requestAnimationFrame(() => {
        document.querySelectorAll(".pos-col .stem[data-h]").forEach((el) => {
          el.style.height = `${el.getAttribute("data-h")}px`;
        });
      });

      $("analyticsSub").textContent = stats.formation
        ? `Formation ${stats.formation} · XI EV ${Number(stats.xi_ev).toFixed(1)} · value ${fmtMoney(stats.squad_value)}`
        : "Expected points shape across the XI.";

      // Moves timeline: realistic FT plan first, then optional chip rebuild, then applied.
      const pendingMoves = plan.transfers || [];
      const chipRebuildMoves = plan.chip_rebuild_transfers || [];
      const execution = data.execution || null;
      const appliedHistory = Array.isArray(data.applied_moves) ? data.applied_moves : [];
      const justApplied = Boolean(execution && execution.saved && pendingMoves.length);
      const appliedList = appliedHistory.length
        ? appliedHistory
        : (justApplied
            ? pendingMoves.map((t) => ({
                ...t,
                applied_at: (data.season && data.season.last_executed_at) || data.generated_at,
                gameweek: (data.deadline && data.deadline.gameweek) || null,
              }))
            : []);
      // If this cycle just saved, don't also list the same batch as "pending".
      const showPending = pendingMoves.length && !(justApplied && appliedList.length);
      const showChipRebuild = chipRebuildMoves.length > 0;
      const chipName = String(plan.chip_to_play || "").replace("freehit", "Free Hit").replace("wildcard", "Wildcard");

      const fmtWhen = (iso) => {
        if (!iso) return "";
        const dt = new Date(iso);
        if (Number.isNaN(dt.getTime())) return String(iso);
        return dt.toLocaleString(undefined, {
          weekday: "short", day: "numeric", month: "short",
          hour: "2-digit", minute: "2-digit",
        });
      };

      const moveBenefit = (t) => {
        const outEv = Number(t.out_ev);
        const inEv = Number(t.in_ev);
        if (!Number.isFinite(outEv) || !Number.isFinite(inEv)) {
          return { cls: "gain-flat", label: "—", delta: null };
        }
        const delta = inEv - outEv;
        let cls = "gain-flat";
        if (delta >= 0.5) cls = "gain-strong";
        else if (delta >= 0.1) cls = "gain-mild";
        else if (delta <= -0.5) cls = "gain-strong-neg";
        else if (delta <= -0.1) cls = "gain-mild-neg";
        const label = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} EV`;
        return { cls, label, delta };
      };

      const renderMoveRow = (t, { dated = false, pending = false } = {}) => {
        const benefit = moveBenefit(t);
        const classes = ["move", benefit.cls, pending ? "pending" : ""].filter(Boolean).join(" ");
        return `
        <div class="${classes}" title="${esc(benefit.delta == null ? "EV delta unavailable" : `Projected EV change ${benefit.label}`)}">
          ${dated && t.applied_at ? `<div class="when">${fmtWhen(t.applied_at)}${t.gameweek != null ? ` · GW${t.gameweek}` : ""}</div>` : ""}
          <div>
            <div class="who">Out · ${esc(t.out_name || "")}${t.out_team ? ` (${esc(t.out_team)})` : ""}</div>
            <div class="meta">${[t.out_position, t.out_price != null ? fmtMoney(t.out_price) : null, t.out_ev != null ? `EV ${Number(t.out_ev).toFixed(2)}` : null].filter(Boolean).join(" · ")}</div>
          </div>
          <div class="arrow">→<span class="delta">${esc(benefit.label)}</span></div>
          <div>
            <div class="who">In · ${esc(t.in_name || "")}${t.in_team ? ` (${esc(t.in_team)})` : ""}</div>
            <div class="meta">${[t.in_position, t.in_price != null ? fmtMoney(t.in_price) : null, t.in_ev != null ? `EV ${Number(t.in_ev).toFixed(2)}` : null].filter(Boolean).join(" · ")}</div>
          </div>
        </div>`;
      };
      function renderAppliedMoves(list) {
        const preview = 5;
        const visible = list.slice(0, preview);
        const rest = list.slice(preview);
        let html = visible.map((t) => renderMoveRow(t, { dated: true })).join("");
        if (rest.length) {
          html += `<button type="button" class="moves-toggle" id="movesExpand">Show ${rest.length} more</button>`;
          html += `<div id="movesExtra" hidden>${rest.map((t) => renderMoveRow(t, { dated: true })).join("")}</div>`;
        }
        return html;
      }

      $("movesTitle").textContent = "Moves";
      const bits = [];
      if (showPending) bits.push(`${pendingMoves.length} under consideration (1 FT)`);
      if (showChipRebuild) bits.push(`${chipRebuildMoves.length} ${chipName || "chip"} sketch`);
      if (appliedList.length) bits.push(`${appliedList.length} applied`);
      $("movesSub").textContent = bits.length
        ? `${bits.join(" · ")} · coloured by projected EV gain/loss`
        : "No transfers in the latest 1-FT plan — live squad already matches the optimiser.";

      let movesHtml = "";
      if (showPending) {
        movesHtml += `<div class="moves-block-label">Under consideration</div>`;
        movesHtml += pendingMoves.map((t) => renderMoveRow(t, { pending: true })).join("");
      }
      if (showChipRebuild) {
        movesHtml += `<div class="moves-block-label">${esc(chipName || "Chip")} sketch (not the 1-FT plan)</div>`;
        movesHtml += chipRebuildMoves.map((t) => renderMoveRow(t, { pending: true })).join("");
      }
      if (appliedList.length) {
        movesHtml += `<div class="moves-block-label">Applied</div>`;
        movesHtml += renderAppliedMoves(appliedList);
      }
      if (!movesHtml) {
        movesHtml = `<div class="empty">No transfers recommended or applied yet.</div>`;
      }
      $("movesList").innerHTML = movesHtml;
      const btn = $("movesExpand");
      if (btn) {
        btn.addEventListener("click", () => {
          const extra = $("movesExtra");
          const open = !extra.hidden;
          extra.hidden = open;
          const restN = appliedList.length - 5;
          btn.textContent = open ? `Show ${restN} more` : "Show less";
        });
      }

      if (showPending) {
        setSectionTitle("xiTitle", "", "Target lineup");
        $("xiSub").textContent =
          `After the transfers under consideration · objective ${plan.objective != null ? Number(plan.objective).toFixed(2) : "—"} · ${stats.formation || "—"}`;
      } else {
        setSectionTitle("xiTitle", "", "Current lineup");
        $("xiSub").textContent =
          `Your 15 with the bot’s start / bench / captain · objective ${plan.objective != null ? Number(plan.objective).toFixed(2) : "—"} · ${stats.formation || "—"}`;
      }
      const squadBits = [];
      if (stats.formation) squadBits.push(stats.formation);
      if (showPending) squadBits.push(`${pendingMoves.length} transfer${pendingMoves.length === 1 ? "" : "s"} pending`);
      else if (appliedList.length) squadBits.push(`${appliedList.length} applied`);
      if (stats.xi_ev != null) squadBits.push(`XI EV ${Number(stats.xi_ev).toFixed(1)}`);
      $("squadSub").textContent = squadBits.length
        ? squadBits.join(" · ")
        : "Lineup, transfers, and EV shape in one place.";
      $("pitch").innerHTML = renderPitch(
        plan.starting_xi || [],
        plan.captain_id,
        plan.vice_captain_id,
      );

      const bench = plan.bench || [];
      $("bench").innerHTML = bench.length
        ? bench.map((p) => {
            const cls = ["player", p.captain ? "captain" : "", p.vice_captain ? "vice" : ""].filter(Boolean).join(" ");
            const label = [p.web_name || p.id, p.team, p.position].filter(Boolean).join(", ");
            return `
          <button type="button" class="${cls}" data-player-id="${esc(p.id)}" aria-label="${esc(label)}">
            <div class="pos">${esc(p.position || "")}</div>
            ${playerPhotoHtml(p, { className: "face" })}
            <div class="name">${esc(p.web_name || p.id)}${p.team ? `<span class="team">${esc(p.team)}</span>` : ""}</div>
            <div class="ev">${p.ev != null ? Number(p.ev).toFixed(2) : "—"}</div>
            <div class="price">${fmtMoney(p.price)}</div>
          </button>`;
          }).join("")
        : `<div class="empty">No bench data yet.</div>`;

      const roster = [...(plan.starting_xi || []), ...bench];
      const byId = new Map(roster.map((p) => [Number(p.id), p]));
      bindPlayerClicks($("pitch"), byId);
      bindPlayerClicks($("bench"), byId);

      const season = data.season || {};
      $("season").innerHTML = [
        metric("Last recommended", season.last_recommended_gw != null ? `GW${season.last_recommended_gw}` : "—"),
        metric("Last executed", season.last_executed_gw != null ? `GW${season.last_executed_gw}` : "—",
          season.last_execute_ok == null ? "" : (season.last_execute_ok ? "OK" : "Failed")),
        metric("Applied moves", String(stats.applied_moves_count ?? 0),
          stats.applied_last_at ? fmtWhen(stats.applied_last_at) : ""),
        metric("Auth left", s.auth_seconds_remaining != null ? fmtHours(s.auth_seconds_remaining / 3600) : "—"),
        metric("Run lock", s.run_lock_held ? "Busy" : "Idle"),
        metric("Signals", (stats.signals && stats.signals.count != null) ? String(stats.signals.count) : "—",
          stats.signals && stats.signals.excluded != null ? `${stats.signals.excluded} excluded` : ""),
      ].join("");

      const intel = data.intel || {};
      const isum = intel.summary || {};
      const news = Array.isArray(intel.news) ? intel.news : [];
      const preseason = Array.isArray(intel.preseason) ? intel.preseason : [];
      const lineups = Array.isArray(intel.predicted_lineups) ? intel.predicted_lineups : [];

      $("intelSub").textContent = isum.updated_at
        ? `Updated ${fmtWhen(isum.updated_at)}${isum.model ? ` · ${isum.model}` : ""}${isum.kg_validated ? " · KG validated" : ""}${(isum.news_press != null || isum.news_fpl != null) ? ` · ${isum.news_press ?? 0} press / ${isum.news_fpl ?? 0} FPL` : (news.length > 5 ? ` · top 5 of ${news.length}` : "")}.`
        : "Player availability signals from news ingest, with source attribution.";

      $("intelMetrics").innerHTML = [
        metric("Press", String(isum.news_press ?? "—"), "Headlines / heuristics"),
        metric("FPL official", String(isum.news_fpl ?? "—"), `${isum.excluded ?? 0} excluded`),
        metric("Preseason", String(isum.preseason ?? preseason.length), "role / form priors"),
        metric(
          "Squad mins",
          (isum.squad_minutes_covered != null && isum.squad_minutes_owned != null)
            ? `${isum.squad_minutes_covered}/${isum.squad_minutes_owned}`
            : "—",
          isum.squad_minutes_missing != null ? `${isum.squad_minutes_missing} missing` : "owned coverage",
        ),
      ].join("");

      const pulseNews = $("pulseNews");
      if (pulseNews) {
        const newsRank = (n) => {
          const kind = String(n.type || "news").toLowerCase();
          const sev = Number(n.severity) || 0;
          let tier = 5;
          if (kind === "injury" || kind === "doubt") tier = 0;
          else if (kind === "suspension") tier = 1;
          else if (kind === "positive" || kind === "start" || kind === "bench") tier = 2;
          else if (kind === "transfer") tier = n.exclude ? 6 : 4;
          else tier = 3;
          return tier * 10 - sev;
        };
        const ranked = [...news].sort((a, b) => newsRank(a) - newsRank(b));
        const top = ranked.slice(0, 3);
        if (!top.length) {
          pulseNews.innerHTML = `
            <div class="pulse-news-head">
              <h3>Recent news</h3>
              <button type="button" class="pulse-news-link" data-open-view="news">Open News</button>
            </div>
            <div class="empty">No player signals yet.</div>`;
        } else {
          const sevChip = (sev) => {
            const v = Math.max(0, Math.min(1, Number(sev) || 0));
            const cls = v >= 0.7 ? "high" : v >= 0.35 ? "mid" : "low";
            const label = v >= 0.7 ? "High" : v >= 0.35 ? "Mid" : "Low";
            return `<span class="sev-chip ${cls}" title="Severity ${v.toFixed(2)}">${label}</span>`;
          };
          const rows = top.map((n) => {
            const name = [n.first_name, n.web_name || n.second_name].filter(Boolean).join(" ")
              || n.web_name || "Unknown";
            const kind = String(n.type || "news").replace(/_/g, " ");
            const blurb = n.summary || n.note || n.reason || n.article_title || n.fpl_news || "Signal update";
            return `<div class="pulse-news-row">
              <div class="row-top">
                <span class="tag">${esc(kind)}</span>
                ${sevChip(n.severity)}
                ${n.exclude ? `<span class="tag exclude">exclude</span>` : ""}
              </div>
              <div class="who">${esc(name)}${n.team ? ` · ${esc(n.team)}` : ""}</div>
              <div class="blurb">${esc(String(blurb).slice(0, 140))}${String(blurb).length > 140 ? "…" : ""}</div>
            </div>`;
          }).join("");
          const more = news.length > top.length ? ` · ${news.length - top.length} more` : "";
          pulseNews.innerHTML = `
            <div class="pulse-news-head">
              <h3>Recent news${more ? `<span style="font-size:0.75em;color:var(--muted);font-weight:500">${esc(more)}</span>` : ""}</h3>
              <button type="button" class="pulse-news-link" data-open-view="news">Open News</button>
            </div>
            ${rows}`;
        }
        pulseNews.querySelectorAll("[data-open-view]").forEach((el) => {
          el.addEventListener("click", () => {
            if (typeof window.showDashboardView === "function") {
              window.showDashboardView(el.getAttribute("data-open-view"));
            }
          });
        });
      }

      const renderSignalRow = (n) => {
        const sev = Math.max(0, Math.min(1, Number(n.severity) || 0));
        const conf = n.confidence != null ? Number(n.confidence) : null;
        const kind = String(n.type || "news").toLowerCase();
        const name = [n.first_name, n.web_name || n.second_name].filter(Boolean).join(" ")
          || n.web_name || "Unknown";
        const via = [n.article_source, n.source].filter(Boolean).map((s) => String(s).replace(/_/g, " "));
        const viaLabel = via.length ? via.join(" · ") : "";
        const sourceBlock = n.article_link
          ? `<div class="source">Via ${esc(viaLabel || "article")} · <a href="${esc(n.article_link)}" target="_blank" rel="noopener">${esc(n.article_title || "Open source")}</a></div>`
          : (n.article_title
            ? `<div class="source">Via ${esc(viaLabel || "article")} · ${esc(n.article_title)}</div>`
            : (viaLabel ? `<div class="source">Via ${esc(viaLabel)}</div>` : ""));
        return `<div class="signal">
          <div class="head">
            <span class="who">${esc(name)}</span>
            ${n.team_short ? `<span class="team">${esc(n.team_short)}</span>` : ""}
            <span class="tag ${esc(kind)}">${esc(kind.replace("_", " "))}</span>
            ${n.exclude ? `<span class="tag exclude">exclude</span>` : ""}
          </div>
          <div class="sev ${sev < 0.35 ? "ok" : ""}">
            <span>sev</span>
            <div class="track"><span style="width:${Math.round(sev * 100)}%"></span></div>
            <span>${sev.toFixed(2)}${conf != null ? ` · conf ${conf.toFixed(2)}` : ""}</span>
          </div>
          ${n.note ? `<div class="note">${esc(n.note)}</div>` : ""}
          ${n.evidence ? `<div class="evidence">“${esc(n.evidence)}”</div>` : ""}
          ${n.fpl_news ? `<div class="evidence">FPL: ${esc(n.fpl_news)}${n.fpl_status ? ` (${esc(n.fpl_status)})` : ""}</div>` : ""}
          ${sourceBlock}
        </div>`;
      };

      const newsPreview = 5;
      if (!news.length) {
        $("intelSignals").innerHTML = `<div class="empty">No player signals yet.</div>`;
      } else {
        const visible = news.slice(0, newsPreview);
        const rest = news.slice(newsPreview);
        let html = visible.map(renderSignalRow).join("");
        if (rest.length) {
          html += `<button type="button" class="moves-toggle" id="intelExpand">Show ${rest.length} more</button>`;
          html += `<div id="intelExtra" hidden>${rest.map(renderSignalRow).join("")}</div>`;
        }
        $("intelSignals").innerHTML = html;
        const intelBtn = $("intelExpand");
        if (intelBtn) {
          intelBtn.addEventListener("click", () => {
            const extra = $("intelExtra");
            const open = !extra.hidden;
            extra.hidden = open;
            intelBtn.textContent = open ? `Show ${rest.length} more` : "Show less";
          });
        }
      }

      if (preseason.length) {
        $("intelPreseasonWrap").hidden = false;
        const psPreview = 5;
        const renderPre = (p) => {
          const boost = p.boost != null ? Number(p.boost) : null;
          return `<div class="signal">
            <div class="head">
              <span class="who">${esc(p.web_name || "—")}</span>
              <span class="tag ${(p.role || "unknown").toLowerCase()}">${esc(p.role || "unknown")}</span>
              ${boost != null ? `<span class="tag">+${boost.toFixed(2)} EV</span>` : ""}
            </div>
            <div class="sev ok">
              <span>mins</span>
              <div class="track"><span style="width:${Math.min(100, Math.round((Number(p.minutes) || 0) / 90 * 100))}%"></span></div>
              <span>${p.minutes ?? 0} · ${p.starts ?? 0} starts</span>
            </div>
            ${p.note ? `<div class="note">${esc(p.note)}</div>` : ""}
          </div>`;
        };
        const psVisible = preseason.slice(0, psPreview);
        const psRest = preseason.slice(psPreview);
        let psHtml = psVisible.map(renderPre).join("");
        if (psRest.length) {
          psHtml += `<button type="button" class="moves-toggle" id="intelPreExpand">Show ${psRest.length} more</button>`;
          psHtml += `<div id="intelPreExtra" hidden>${psRest.map(renderPre).join("")}</div>`;
        }
        $("intelPreseason").innerHTML = psHtml;
        const psBtn = $("intelPreExpand");
        if (psBtn) {
          psBtn.addEventListener("click", () => {
            const extra = $("intelPreExtra");
            const open = !extra.hidden;
            extra.hidden = open;
            psBtn.textContent = open ? `Show ${psRest.length} more` : "Show less";
          });
        }
      } else {
        $("intelPreseasonWrap").hidden = true;
        $("intelPreseason").innerHTML = "";
      }

      if (lineups.length) {
        $("intelLineupsWrap").hidden = false;
        const lineupPreview = 5;
        const renderLineupRow = (l) => {
          const role = String(l.role || "start").toLowerCase();
          const conf = l.confidence != null ? Number(l.confidence) : null;
          const mins = l.predicted_minutes != null ? Number(l.predicted_minutes) : null;
          return `<div class="signal">
            <div class="head">
              <span class="who">${esc(l.web_name || "—")}</span>
              ${l.team_short ? `<span class="team">${esc(l.team_short)}</span>` : ""}
              <span class="tag ${esc(role)}">${esc(role)}</span>
              ${mins != null && !Number.isNaN(mins) ? `<span class="tag">${esc(String(mins))}m</span>` : ""}
              ${l.gameweek != null ? `<span class="tag">GW${esc(l.gameweek)}</span>` : ""}
            </div>
            ${conf != null ? `<div class="sev ok"><span>conf</span><div class="track"><span style="width:${Math.round(Math.max(0, Math.min(1, conf)) * 100)}%"></span></div><span>${conf.toFixed(2)}</span></div>` : ""}
            ${l.note ? `<div class="note">${esc(l.note)}</div>` : ""}
          </div>`;
        };
        const luVisible = lineups.slice(0, lineupPreview);
        const luRest = lineups.slice(lineupPreview);
        let luHtml = luVisible.map(renderLineupRow).join("");
        if (luRest.length) {
          luHtml += `<button type="button" class="moves-toggle" id="intelLineupsExpand">Show ${luRest.length} more</button>`;
          luHtml += `<div id="intelLineupsExtra" hidden>${luRest.map(renderLineupRow).join("")}</div>`;
        }
        $("intelLineups").innerHTML = luHtml;
        const luBtn = $("intelLineupsExpand");
        if (luBtn) {
          luBtn.addEventListener("click", () => {
            const extra = $("intelLineupsExtra");
            const open = !extra.hidden;
            extra.hidden = open;
            luBtn.textContent = open ? `Show ${luRest.length} more` : "Show less";
          });
        }
      } else {
        $("intelLineupsWrap").hidden = true;
        $("intelLineups").innerHTML = "";
      }

      const minutesCheck = intel.squad_minutes_check || {};
      const minutesPlayers = Array.isArray(minutesCheck.players) ? minutesCheck.players : [];
      if (minutesPlayers.length) {
        $("intelMinutesWrap").hidden = false;
        const covered = minutesCheck.covered ?? minutesPlayers.filter((p) => p.covered).length;
        const owned = minutesCheck.owned ?? minutesPlayers.length;
        const missing = minutesCheck.missing_count ?? Math.max(0, owned - covered);
        $("intelMinutesSub").textContent =
          `${covered}/${owned} owned players have minutes`
          + (minutesCheck.news_covered != null ? ` · ${minutesCheck.news_covered} from news` : "")
          + (minutesCheck.fpl_implied_covered != null ? ` · ${minutesCheck.fpl_implied_covered} FPL proxy` : "")
          + (missing ? ` · ${missing} missing` : "")
          + (minutesCheck.gameweek != null ? ` · GW${minutesCheck.gameweek}` : "")
          + ".";
        const renderMins = (p) => {
          const role = p.role ? String(p.role).toLowerCase() : "none";
          const mins = p.predicted_minutes != null ? Number(p.predicted_minutes) : null;
          const conf = p.confidence != null ? Number(p.confidence) : null;
          const src = String(p.source || "");
          const srcLabel = src === "fpl_implied" ? "fpl"
            : (src === "heuristic_mention" ? "mention"
            : (src === "transfer_lineup" ? "transfer"
            : (src === "depth_model" ? "depth"
            : (src.replace(/_lineup$/, "") || ""))));
          return `<div class="signal">
            <div class="head">
              <span class="who">${esc(p.web_name || "—")}</span>
              ${p.team_short ? `<span class="team">${esc(p.team_short)}</span>` : ""}
              <span class="tag ${p.covered ? esc(role) : "missing"}">${p.covered ? esc(role) : "no coverage"}</span>
              ${mins != null && !Number.isNaN(mins) ? `<span class="tag">${esc(String(mins))}m</span>` : ""}
              ${srcLabel ? `<span class="tag">${esc(srcLabel)}</span>` : ""}
            </div>
            ${conf != null ? `<div class="sev ok"><span>conf</span><div class="track"><span style="width:${Math.round(Math.max(0, Math.min(1, conf)) * 100)}%"></span></div><span>${conf.toFixed(2)}</span></div>` : ""}
            ${p.note ? `<div class="note">${esc(p.note)}</div>` : ""}
          </div>`;
        };
        // Show missing first, then covered with lowest minutes.
        const ordered = [...minutesPlayers].sort((a, b) => {
          if (!!a.covered !== !!b.covered) return a.covered ? 1 : -1;
          const am = a.predicted_minutes == null ? 999 : Number(a.predicted_minutes);
          const bm = b.predicted_minutes == null ? 999 : Number(b.predicted_minutes);
          return am - bm;
        });
        const preview = 8;
        const visible = ordered.slice(0, preview);
        const rest = ordered.slice(preview);
        let html = visible.map(renderMins).join("");
        if (rest.length) {
          html += `<button type="button" class="moves-toggle" id="intelMinutesExpand">Show ${rest.length} more</button>`;
          html += `<div id="intelMinutesExtra" hidden>${rest.map(renderMins).join("")}</div>`;
        }
        $("intelMinutes").innerHTML = html;
        const btn = $("intelMinutesExpand");
        if (btn) {
          btn.addEventListener("click", () => {
            const extra = $("intelMinutesExtra");
            const open = !extra.hidden;
            extra.hidden = open;
            btn.textContent = open ? `Show ${rest.length} more` : "Show less";
          });
        }
      } else {
        $("intelMinutesWrap").hidden = true;
        $("intelMinutes").innerHTML = "";
      }

      $("logs").textContent = ((data.logs || {}).season || []).join("\n") || "No season log yet.";
      const logsEl = $("logs");
      const logsBtn = $("logsToggle");
      if (logsEl) logsEl.classList.toggle("is-collapsed", !logsOpen);
      if (logsBtn) {
        logsBtn.setAttribute("aria-expanded", logsOpen ? "true" : "false");
        logsBtn.textContent = logsOpen ? "Hide daemon log" : "Show daemon log";
      }
      $("foot").textContent = `Updated ${new Date(data.generated_at).toLocaleString()} · summary ${data.summary_mtime ? new Date(data.summary_mtime).toLocaleString() : "n/a"} · refreshes every 20s`;
      polishMotion();
      requestAnimationFrame(() => requestAnimationFrame(syncMovesPaneHeight));

      if (typeof window.setFxMood === "function") {
        const intelSum = (data.intel && data.intel.summary) || {};
        const sigKey = [
          intelSum.total,
          intelSum.news_press,
          intelSum.news_fpl,
          data.signals_mtime || "",
          (data.intel && data.intel.ingest && data.intel.ingest.updated_at) || "",
        ].join("|");
        window.setFxMood({
          deadlineHours: d.hours_to_deadline,
          authOk: s.auth_ok,
          authSeconds: s.auth_seconds_remaining,
          signalsKey: sigKey,
        });
      }
    }

    async function tick() {
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        const data = await res.json();
        render(data);
      } catch (err) {
        $("lede").textContent = `Dashboard could not load bot state (${err}).`;
      }
    }

    tick();
    setInterval(tick, liteFx ? 45000 : 20000);

    (function initNav() {
      const nav = $("siteNav");
      const links = Array.from(document.querySelectorAll("[data-nav]"));
      const indicator = $("navIndicator");
      const jump = $("navJump");
      const scroller = nav && nav.querySelector(".nav-scroller");
      const brand = document.querySelector(".nav-brand");
      const stage = $("stagePanels") || $("stage");
      if (!nav || !links.length) return;

      const sectionIds = links
        .map((a) => (a.getAttribute("href") || "").replace(/^#/, ""))
        .filter(Boolean);
      const sections = sectionIds
        .map((id) => document.getElementById(id))
        .filter(Boolean);

      let activeId = sections.find((s) => s.classList.contains("is-active"))?.id
        || (sections[0] ? sections[0].id : "pulse");
      try {
        const saved = sessionStorage.getItem(VIEW_KEY);
        if (saved && sectionIds.includes(saved)) activeId = saved;
        else if (saved === "hub") activeId = "pulse";
      } catch (_) { /* ignore */ }

      function moveIndicator(link) {
        if (!indicator || !link) return;
        indicator.style.width = `${link.offsetWidth}px`;
        indicator.style.transform = `translateX(${link.offsetLeft}px)`;
        if (scroller) {
          const left = link.offsetLeft - 24;
          const right = link.offsetLeft + link.offsetWidth + 24;
          if (left < scroller.scrollLeft) scroller.scrollLeft = Math.max(0, left);
          else if (right > scroller.scrollLeft + scroller.clientWidth) {
            scroller.scrollLeft = right - scroller.clientWidth;
          }
        }
      }

      function showView(id, { replaceHistory = true } = {}) {
        if (!id || !sectionIds.includes(id)) return;
        activeId = id;

        sections.forEach((sec) => {
          const on = sec.id === id;
          sec.classList.toggle("is-active", on);
          if (on) sec.scrollTop = 0;
        });

        links.forEach((a) => {
          const on = a.getAttribute("href") === `#${id}`;
          a.classList.toggle("is-active", on);
          if (on) moveIndicator(a);
        });

        if (jump) {
          const opt = `#${id}`;
          if ([...jump.options].some((o) => o.value === opt)) jump.value = opt;
        }

        try { sessionStorage.setItem(VIEW_KEY, id); } catch (_) { /* ignore */ }

        if (replaceHistory) {
          try {
            history.replaceState(null, "", location.pathname + location.search);
          } catch (_) { /* ignore */ }
        }
      }

      function viewIndex(id) {
        return sectionIds.indexOf(id);
      }

      function shiftView(delta) {
        const i = viewIndex(activeId);
        if (i < 0) return;
        const next = sectionIds[i + delta];
        if (next) showView(next);
      }

      links.forEach((a) => {
        a.addEventListener("click", (ev) => {
          ev.preventDefault();
          const id = (a.getAttribute("href") || "").replace(/^#/, "");
          showView(id);
        });
      });

      if (brand) {
        brand.addEventListener("click", (ev) => {
          ev.preventDefault();
          showView("pulse");
        });
      }

      if (jump) {
        jump.addEventListener("change", () => {
          const href = jump.value;
          if (!href) return;
          if (href.startsWith("/") && !href.startsWith("/#")) {
            window.location.href = href;
            return;
          }
          showView(href.replace(/^#/, ""));
        });
      }

      // Desktop: ← / → between adjacent panels (ignore when typing in controls).
      window.addEventListener("keydown", (ev) => {
        if (ev.defaultPrevented) return;
        const tag = (ev.target && ev.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || ev.target?.isContentEditable) {
          return;
        }
        if (ev.key === "ArrowLeft") {
          ev.preventDefault();
          shiftView(-1);
        } else if (ev.key === "ArrowRight") {
          ev.preventDefault();
          shiftView(1);
        }
      });

      // Mobile: horizontal swipe on the stage switches panels.
      if (stage) {
        let startX = 0;
        let startY = 0;
        let tracking = false;
        stage.addEventListener("touchstart", (ev) => {
          if (!ev.touches || ev.touches.length !== 1) return;
          tracking = true;
          startX = ev.touches[0].clientX;
          startY = ev.touches[0].clientY;
        }, { passive: true });
        stage.addEventListener("touchend", (ev) => {
          if (!tracking) return;
          tracking = false;
          const t = ev.changedTouches && ev.changedTouches[0];
          if (!t) return;
          const dx = t.clientX - startX;
          const dy = t.clientY - startY;
          if (Math.abs(dx) < 56 || Math.abs(dx) < Math.abs(dy) * 1.35) return;
          if (dx < 0) shiftView(1);
          else shiftView(-1);
        }, { passive: true });
        stage.addEventListener("touchcancel", () => { tracking = false; }, { passive: true });
      }

      window.addEventListener("resize", () => {
        const cur = links.find((a) => a.classList.contains("is-active")) || links[0];
        moveIndicator(cur);
      });

      showView(activeId, { replaceHistory: false });
      window.showDashboardView = showView;
    })();

    function initNetworkGraph() {
      const layer = $("fxLayer") || document.querySelector(".fx-layer");
      const nearSvg = $("fxNetwork");
      const farSvg = $("fxNetworkFar");
      if (!layer) return;

      // Phones / reduced-motion: no SVG constellation, no rAF loop.
      if (liteFx || reduceMotion) {
        if (nearSvg) nearSvg.innerHTML = "";
        if (farSvg) farSvg.innerHTML = "";
        window.setFxMood = function setFxMoodLite({ deadlineHours, authOk, authSeconds } = {}) {
          const hours = deadlineHours == null ? 999 : Number(deadlineHours);
          let urgency = 0;
          if (hours > 0 && hours < 48) urgency = Math.min(1, (48 - hours) / 48);
          if (hours > 0 && hours <= 2) urgency = Math.max(urgency, 0.75);
          if (hours > 0 && hours <= 0.5) urgency = 1;
          layer.dataset.mood = authOk === false ? "bad" : (authSeconds != null && authSeconds < 3600 ? "warn" : "ok");
          layer.dataset.urgency = urgency >= 0.65 ? "high" : urgency >= 0.3 ? "mid" : "low";
        };
        return;
      }

      if (!nearSvg) return;

      const mood = {
        urgency: 0,       // 0..1 from deadline proximity
        authWarn: false,
        authBad: false,
        pulseMul: 1,
        packetMul: 1,
        lastSignalsKey: null,
        bloomUntil: 0,
      };

      function buildGraph(svg, opts) {
        const W = 1200;
        const WORLD_H = opts.worldH || 3200;
        const VIEW_H = opts.viewH || 900;
        const count = opts.nodeCount;
        const hubs = [];
        for (let i = 0; i < count; i++) {
          // Spread nodes through a tall world so scrolling pans into new clusters.
          const col = (i * 197 + opts.seed * 13) % (W - 100) + 50;
          const row = (i * 173 + opts.seed * 41) % (WORLD_H - 120) + 60;
          hubs.push([col, row]);
        }
        // A few denser “constellation bands” down the page.
        for (let band = 0; band < 5; band++) {
          const cy = 280 + band * (WORLD_H / 5);
          for (let k = 0; k < (opts.far ? 4 : 7); k++) {
            hubs.push([
              140 + ((band * 90 + k * 150 + opts.seed * 11) % (W - 280)),
              cy + ((k * 47 + opts.seed) % 160) - 80,
            ]);
          }
        }
        const nodes = hubs.map(([x, y], i) => ({
          ox: x,
          oy: y,
          x, y,
          r: opts.far
            ? (i % 5 === 0 ? 2.8 : 1.4)
            : (i % 4 === 0 ? 3.8 : (i % 3 === 0 ? 2.6 : 2.0)),
          hub: opts.far ? i % 5 === 0 : i % 4 === 0,
          phase: (i * 0.73 + opts.seed) % (Math.PI * 2),
          speed: (opts.far ? 0.22 : 0.35) + (i % 5) * 0.08,
          ampX: (opts.far ? 14 : 9) + (i % 6) * 1.8,
          ampY: (opts.far ? 12 : 7) + (i % 7) * 1.5,
          el: null,
        }));

        const edgePairs = [];
        const linkR = opts.far ? 300 : 260;
        for (let i = 0; i < nodes.length; i++) {
          const best = [];
          for (let j = 0; j < nodes.length; j++) {
            if (i === j) continue;
            const dx = nodes[i].ox - nodes[j].ox;
            const dy = nodes[i].oy - nodes[j].oy;
            const d2 = dx * dx + dy * dy;
            if (d2 < linkR * linkR) best.push({ j, d2 });
          }
          best.sort((a, b) => a.d2 - b.d2);
          best.slice(0, nodes[i].hub ? (opts.far ? 3 : 5) : (opts.far ? 2 : 3)).forEach(({ j }) => {
            if (i < j) edgePairs.push([i, j]);
          });
        }

        const edges = edgePairs.map(([a, b], idx) => ({
          a, b,
          live: idx % (opts.far ? 5 : 4) === 0,
          el: null,
          hotUntil: 0,
        }));
        const liveEdges = edges.filter((e) => e.live);
        const packets = opts.packets
          ? Array.from({ length: opts.packets }, (_, i) => ({
              edge: i % Math.max(1, liveEdges.length),
              t: (i * 0.17) % 1,
              speed: 0.12 + (i % 3) * 0.04,
              el: null,
            }))
          : [];

        let html = "";
        edges.forEach((e, idx) => {
          const live = e.live ? " is-live" : "";
          const dur = (opts.far ? 7 : 3.5) + (idx % 5) * 0.7;
          html += `<line class="edge${live}" data-ei="${idx}" style="animation-duration:${dur}s;animation-delay:-${(idx % 9) * 0.4}s" />`;
        });
        packets.forEach((_, i) => {
          html += `<circle class="packet" data-pi="${i}" r="${opts.far ? 1.6 : 2.4}" />`;
        });
        nodes.forEach((n, i) => {
          html += `<circle class="node${n.hub ? " is-hub" : ""}" data-ni="${i}" r="${n.r}" />`;
        });
        svg.innerHTML = html;
        svg.setAttribute("viewBox", `0 0 ${W} ${VIEW_H}`);
        edges.forEach((e, i) => { e.el = svg.querySelector(`[data-ei="${i}"]`); });
        nodes.forEach((n, i) => { n.el = svg.querySelector(`[data-ni="${i}"]`); });
        packets.forEach((p, i) => { p.el = svg.querySelector(`[data-pi="${i}"]`); });

        return {
          svg, nodes, edges, liveEdges, packets,
          lean: opts.lean,
          panFactor: opts.panFactor == null ? 1 : opts.panFactor,
          travelScreens: opts.travelScreens == null ? 1.05 : opts.travelScreens,
          worldW: W,
          worldH: WORLD_H,
          viewH: VIEW_H,
          viewW: W,
        };
      }

      const far = farSvg
        ? buildGraph(farSvg, {
            seed: 3, nodeCount: 34, far: true, packets: 0, lean: 0.4, panFactor: 0.7,
            travelScreens: 0.7, worldH: 2600, viewH: 900,
          })
        : null;
      const near = buildGraph(nearSvg, {
        seed: 7, nodeCount: 52, far: false, packets: 7, lean: 1, panFactor: 1,
        travelScreens: 1.05, worldH: 2400, viewH: 900,
      });

      function paintStatic(g) {
        if (!g) return;
        g.nodes.forEach((n) => {
          n.el.setAttribute("cx", n.ox);
          n.el.setAttribute("cy", n.oy);
        });
        g.edges.forEach((e) => {
          e.el.setAttribute("x1", g.nodes[e.a].ox);
          e.el.setAttribute("y1", g.nodes[e.a].oy);
          e.el.setAttribute("x2", g.nodes[e.b].ox);
          e.el.setAttribute("y2", g.nodes[e.b].oy);
        });
        g.packets.forEach((p) => { if (p.el) p.el.style.display = "none"; });
        g.svg.setAttribute("viewBox", `0 0 ${g.viewW} ${g.viewH}`);
      }

      window.setFxMood = function setFxMood({ deadlineHours, authOk, authSeconds, signalsKey } = {}) {
        const hours = deadlineHours == null ? 999 : Number(deadlineHours);
        let urgency = 0;
        if (hours > 0 && hours < 48) urgency = Math.min(1, (48 - hours) / 48);
        if (hours > 0 && hours <= 2) urgency = Math.max(urgency, 0.75);
        if (hours > 0 && hours <= 0.5) urgency = 1;
        mood.urgency = urgency;
        mood.authBad = authOk === false;
        mood.authWarn = !mood.authBad && authSeconds != null && authSeconds < 3600;
        mood.pulseMul = 1 + urgency * 1.8;
        mood.packetMul = 1 + urgency * 1.4 + (mood.authWarn || mood.authBad ? 0.35 : 0);

        const moodKey = mood.authBad ? "bad" : mood.authWarn ? "warn" : "ok";
        const urgKey = urgency >= 0.65 ? "high" : urgency >= 0.3 ? "mid" : "low";
        layer.dataset.mood = moodKey;
        layer.dataset.urgency = urgKey;

        if (signalsKey != null && mood.lastSignalsKey != null && signalsKey !== mood.lastSignalsKey) {
          mood.bloomUntil = performance.now() + 1400;
          layer.classList.add("is-bloom");
          setTimeout(() => layer.classList.remove("is-bloom"), 1500);
        }
        if (signalsKey != null) mood.lastSignalsKey = signalsKey;
      };

      if (reduceMotion) {
        paintStatic(far);
        paintStatic(near);
        return;
      }

      let scrollY = window.scrollY || 0;
      let targetScroll = scrollY;
      let pointerX = 0.5;
      let pointerY = 0.35;
      let targetPX = pointerX;
      let targetPY = pointerY;
      let lastHot = 0;
      let scrollProgress = 0;
      let running = true;
      let last = performance.now();

      window.addEventListener("scroll", () => { targetScroll = window.scrollY || 0; }, { passive: true });
      window.addEventListener("pointermove", (e) => {
        const w = window.innerWidth || 1;
        const h = window.innerHeight || 1;
        targetPX = Math.min(1, Math.max(0, e.clientX / w));
        targetPY = Math.min(1, Math.max(0, e.clientY / h));
      }, { passive: true });

      document.addEventListener("visibilitychange", () => {
        running = document.visibilityState === "visible";
        if (running) {
          last = performance.now();
          requestAnimationFrame(frame);
        }
      });

      function pageProgress(y) {
        const max = Math.max(1, (document.documentElement.scrollHeight || 1) - (window.innerHeight || 1));
        return Math.min(1, Math.max(0, y / max));
      }

      function stepGraph(g, t, dt, now, leanX, leanY, progress) {
        if (!g) return;
        const urg = mood.urgency;
        // Pan slowly through the tall world: less than one viewport of graph
        // across the whole page, so the background never outruns the content.
        const maxPanY = Math.max(0, g.worldH - g.viewH);
        const travel = g.viewH * (g.travelScreens == null ? 1.05 : g.travelScreens);
        const maxPanX = 48;
        const panY = progress * Math.min(maxPanY, travel) * g.panFactor;
        const panX = leanX * maxPanX * g.lean;
        g.svg.setAttribute(
          "viewBox",
          `${panX.toFixed(2)} ${panY.toFixed(2)} ${g.viewW} ${g.viewH}`
        );
        // Light viewport nudge (kept tiny so nothing ever clips away).
        g.svg.style.transform =
          `translate3d(${(leanX * 10 * g.lean).toFixed(2)}px, ${(leanY * 8 * g.lean).toFixed(2)}px, 0) ` +
          `rotate(${(leanX * 0.9 * g.lean).toFixed(2)}deg)`;

        g.nodes.forEach((n) => {
          const speed = n.speed * (1 + urg * 0.8);
          n.x = n.ox
            + Math.sin(t * speed + n.phase) * n.ampX
            + leanX * (n.hub ? 8 : 4) * g.lean;
          n.y = n.oy
            + Math.cos(t * speed * 0.85 + n.phase * 1.3) * n.ampY
            + leanY * (n.hub ? 6 : 3) * g.lean;
          n.el.setAttribute("cx", n.x.toFixed(2));
          n.el.setAttribute("cy", n.y.toFixed(2));
          if (n.hub) {
            const pulse = 1 + (0.1 + urg * 0.16) * Math.sin(t * (2.2 * mood.pulseMul) + n.phase);
            n.el.setAttribute("r", (n.r * pulse).toFixed(2));
          }
        });
        g.edges.forEach((e) => {
          const a = g.nodes[e.a];
          const b = g.nodes[e.b];
          e.el.setAttribute("x1", a.x.toFixed(2));
          e.el.setAttribute("y1", a.y.toFixed(2));
          e.el.setAttribute("x2", b.x.toFixed(2));
          e.el.setAttribute("y2", b.y.toFixed(2));
          e.el.classList.toggle("is-hot", now < e.hotUntil);
        });
        g.packets.forEach((p) => {
          if (!g.liveEdges.length || !p.el) return;
          p.t += p.speed * mood.packetMul * dt;
          if (p.t >= 1) {
            p.t = 0;
            const prev = g.liveEdges[p.edge];
            if (prev) prev.hotUntil = now + 420;
            p.edge = (p.edge + 1 + Math.floor(Math.random() * 2)) % g.liveEdges.length;
          }
          const e = g.liveEdges[p.edge];
          const a = g.nodes[e.a];
          const b = g.nodes[e.b];
          p.el.setAttribute("cx", (a.x + (b.x - a.x) * p.t).toFixed(2));
          p.el.setAttribute("cy", (a.y + (b.y - a.y) * p.t).toFixed(2));
        });
      }

      function frame(now) {
        if (!running) return;
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        const t = now / 1000;

        scrollY += (targetScroll - scrollY) * 0.1;
        pointerX += (targetPX - pointerX) * 0.06;
        pointerY += (targetPY - pointerY) * 0.06;
        scrollProgress += (pageProgress(scrollY) - scrollProgress) * 0.1;

        const leanX = pointerX - 0.5;
        const leanY = pointerY - 0.5;
        layer.style.transform = "";

        stepGraph(far, t * 0.7, dt, now, leanX, leanY, scrollProgress);
        stepGraph(near, t, dt, now, leanX, leanY, scrollProgress);

        if (now - lastHot > (mood.urgency > 0.5 ? 1100 : 2200)) {
          lastHot = now;
          const e = near.edges[(Math.random() * near.edges.length) | 0];
          e.hotUntil = now + 700;
        }

        if (mood.bloomUntil && now > mood.bloomUntil) {
          mood.bloomUntil = 0;
          layer.classList.remove("is-bloom");
        }

        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }

    initNetworkGraph();
    initSectionReveal();
    initSquadTabs();

    (function initPlayerSheet() {
      const close = () => closePlayerSheet();
      $("playerSheetClose")?.addEventListener("click", close);
      $("playerSheetBackdrop")?.addEventListener("click", close);
      window.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") close();
      });
    })();

    (function initChromeControls() {
      const expandBtn = $("statusExpand");
      const strip = $("statusStrip");
      if (expandBtn && strip) {
        expandBtn.addEventListener("click", () => {
          statusExpanded = !statusExpanded;
          strip.classList.toggle("is-expanded", statusExpanded);
          expandBtn.setAttribute("aria-expanded", statusExpanded ? "true" : "false");
          expandBtn.textContent = statusExpanded ? "Less" : "More";
        });
      }
      const logsBtn = $("logsToggle");
      const logsEl = $("logs");
      if (logsBtn && logsEl) {
        logsBtn.addEventListener("click", () => {
          logsOpen = !logsOpen;
          logsEl.classList.toggle("is-collapsed", !logsOpen);
          logsBtn.setAttribute("aria-expanded", logsOpen ? "true" : "false");
          logsBtn.textContent = logsOpen ? "Hide daemon log" : "Show daemon log";
        });
      }
    })();
  