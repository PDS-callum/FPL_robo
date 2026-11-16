(() => {
  const $ = (id) => document.getElementById(id);

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtKick(iso) {
    if (!iso) return "TBC";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return "TBC";
      return d.toLocaleString(undefined, {
        weekday: "short",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (_) {
      return "TBC";
    }
  }

  function fmtHours(h) {
    if (h == null || Number.isNaN(h)) return "—";
    if (h >= 48) return `${(h / 24).toFixed(1)}d`;
    if (h >= 1) return `${h.toFixed(1)}h`;
    return `${Math.max(0, Math.round(h * 60))}m`;
  }

  function fmtNet(n) {
    const v = Number(n) || 0;
    const abs = Math.abs(v);
    const s = abs >= 1000 ? `${(abs / 1000).toFixed(abs >= 10000 ? 0 : 1)}k` : String(abs);
    return v > 0 ? `+${s}` : v < 0 ? `−${s}` : "0";
  }

  function fmtPriceDelta(tenths) {
    const v = Number(tenths) || 0;
    if (!v) return "£0.0";
    const pounds = (Math.abs(v) / 10).toFixed(1);
    return v > 0 ? `+£${pounds}` : `−£${pounds}`;
  }

  function fdrChip(cell) {
    if (!cell || cell.blank || cell.fdr == null) {
      return `<span class="fdr-chip blank">—</span>`;
    }
    const f = Number(cell.fdr);
    return `<span class="fdr-chip fdr-${f}" title="${esc(cell.label || "")}"><span>${esc(cell.opp || "")}</span><span class="ha">${cell.home ? "H" : "A"}</span></span>`;
  }

  function playerRow(p, i, { valueHtml, barPct, tone } = {}) {
    const owned = p.owned ? " owned" : "";
    const bar = barPct != null
      ? `<div class="bar"><span style="width:${Math.max(4, Math.min(100, barPct))}%"></span></div>`
      : "";
    return `<div class="trow${owned}">
      <span class="rank">${i + 1}</span>
      <div class="who">
        <span class="nm">${esc(p.name)}</span>
        <span class="meta">${esc(p.team || "")} · ${esc(p.pos || "")} · £${Number(p.price || 0).toFixed(1)}</span>
      </div>
      <span class="val ${tone || ""}">${valueHtml}</span>
      ${bar}
    </div>`;
  }

  function trendCard(title, hint, rowsHtml, tone) {
    return `<article class="trend-card tone-${tone || "neutral"}">
      <h3>${esc(title)}</h3>
      <p class="hint">${esc(hint)}</p>
      ${rowsHtml || `<p class="empty">Nothing standing out.</p>`}
    </article>`;
  }

  let hubStatsSort = { key: "form", dir: -1 };
  let hubStatsFilter = { pos: "ALL", q: "", ownedOnly: false };

  function initHubTabs() {
    const tabs = Array.from(document.querySelectorAll("[data-hub-tab]"));
    const panes = Array.from(document.querySelectorAll("[data-hub-pane]"));
    if (!tabs.length || !panes.length) return;

    function showTab(id) {
      tabs.forEach((btn) => {
        const on = btn.getAttribute("data-hub-tab") === id;
        btn.classList.toggle("is-active", on);
        btn.setAttribute("aria-selected", on ? "true" : "false");
      });
      panes.forEach((pane) => {
        pane.classList.toggle("is-active", pane.getAttribute("data-hub-pane") === id);
      });
      try { sessionStorage.setItem("fpl_hub_tab", id); } catch (_) { /* ignore */ }
      if (location.hash.replace(/^#/, "") !== id) {
        try { history.replaceState(null, "", `#${id}`); } catch (_) { /* ignore */ }
      }
    }

    tabs.forEach((btn) => {
      btn.addEventListener("click", () => showTab(btn.getAttribute("data-hub-tab")));
    });

    let start = "overview";
    try {
      const saved = sessionStorage.getItem("fpl_hub_tab");
      if (saved && tabs.some((t) => t.getAttribute("data-hub-tab") === saved)) start = saved;
    } catch (_) { /* ignore */ }
    const hash = (location.hash || "").replace(/^#/, "");
    if (hash && tabs.some((t) => t.getAttribute("data-hub-tab") === hash)) start = hash;
    showTab(start);
  }

  function renderOverview(hub) {
    const el = $("hubOverview");
    if (!el) return;
    const t = hub.trends || {};
    const fdr = hub.fdr || {};

    const maxAbs = (list, key) => {
      let m = 1;
      (list || []).forEach((p) => { m = Math.max(m, Math.abs(Number(p[key]) || 0)); });
      return m;
    };

    const risers = t.price_risers || [];
    const fallers = t.price_fallers || [];
    const tin = t.transfer_in || [];
    const tout = t.transfer_out || [];
    const hot = t.form_hot || [];
    const cold = t.form_cold || [];
    const maxRise = maxAbs(risers, "price_delta");
    const maxFall = maxAbs(fallers, "price_delta");
    const maxTin = maxAbs(tin, "net_transfers");
    const maxTout = maxAbs(tout, "net_transfers");
    const maxHot = maxAbs(hot, "delta_form");
    const maxCold = maxAbs(cold, "delta_form");

    const priceUp = risers.filter((p) => (p.price_delta || 0) > 0).map((p, i) =>
      playerRow(p, i, {
        valueHtml: esc(fmtPriceDelta(p.price_delta)),
        tone: "up",
        barPct: (Math.abs(p.price_delta) / maxRise) * 100,
      })
    ).join("");

    const priceDown = fallers.filter((p) => (p.price_delta || 0) < 0).map((p, i) =>
      playerRow(p, i, {
        valueHtml: esc(fmtPriceDelta(p.price_delta)),
        tone: "down",
        barPct: (Math.abs(p.price_delta) / maxFall) * 100,
      })
    ).join("");

    const transfersIn = tin.filter((p) => (p.net_transfers || 0) > 0).map((p, i) =>
      playerRow(p, i, {
        valueHtml: esc(fmtNet(p.net_transfers)),
        tone: "up",
        barPct: (Math.abs(p.net_transfers) / maxTin) * 100,
      })
    ).join("");

    const transfersOut = tout.filter((p) => (p.net_transfers || 0) < 0).map((p, i) =>
      playerRow(p, i, {
        valueHtml: esc(fmtNet(p.net_transfers)),
        tone: "down",
        barPct: (Math.abs(p.net_transfers) / maxTout) * 100,
      })
    ).join("");

    const formHot = hot.map((p, i) =>
      playerRow(p, i, {
        valueHtml: `form ${Number(p.form || 0).toFixed(1)} <span class="meta">(Δ${p.delta_form > 0 ? "+" : ""}${Number(p.delta_form || 0).toFixed(1)})</span>`,
        tone: "hot",
        barPct: (Math.abs(p.delta_form) / maxHot) * 100,
      })
    ).join("");

    const formCold = cold.map((p, i) =>
      playerRow(p, i, {
        valueHtml: `form ${Number(p.form || 0).toFixed(1)} <span class="meta">(Δ${Number(p.delta_form || 0).toFixed(1)})</span>`,
        tone: "down",
        barPct: (Math.abs(p.delta_form) / maxCold) * 100,
      })
    ).join("");

    const runBlock = (title, hint, rows, tone) => {
      const body = (rows || []).map((r) => `
        <div class="run-row">
          <span class="club">${esc(r.short || r.name || "")}</span>
          <div class="chip-row">${(r.next5 || []).map(fdrChip).join("")}</div>
          <span class="ease">${r.run_ease != null ? esc(r.run_ease) : "—"}</span>
        </div>`).join("");
      return `<article class="trend-card tone-${tone}">
        <h3>${esc(title)}</h3>
        <p class="hint">${esc(hint)}</p>
        <div class="run-list">${body || `<p class="empty">No runs yet.</p>`}</div>
      </article>`;
    };

    const alerts = t.owned_alerts || [];
    const alertHtml = alerts.length
      ? `<div class="alert-list">${alerts.map((p) => `
          <div class="alert">
            <span class="nm">${esc(p.name)}</span>
            <span class="meta">${esc(p.team || "")} · ${esc(p.pos || "")}</span>
            ${(p.flags || []).map((f) => `<span class="flag ${esc(f.kind)}">${esc(f.label)}</span>`).join("")}
            ${p.news ? `<span class="meta">${esc(p.news)}</span>` : ""}
          </div>`).join("")}</div>`
      : `<p class="empty">No squad flags right now — form, price, and fixtures look quiet.</p>`;

    const squadFx = t.squad_fixtures || [];
    const squadFxHtml = squadFx.length
      ? `<div class="squad-fx-list">${squadFx.map((p) => `
          <div class="squad-fx-row">
            <div class="who">
              <span class="nm">${esc(p.name)}</span>
              <span class="meta">${esc(p.pos || "")} · ${esc(p.team || "")} · EP ${Number(p.ep || 0).toFixed(1)}</span>
            </div>
            <div class="chip-row">${(p.next5 || []).map(fdrChip).join("")}</div>
            <span class="ease">${p.run_ease != null ? esc(p.run_ease) : "—"}</span>
          </div>`).join("")}</div>`
      : `<p class="empty">No owned squad loaded yet.</p>`;

    const captains = (t.captain_picks || []).map((p, i) =>
      playerRow(p, i, {
        valueHtml: `EP ${Number(p.ep || 0).toFixed(1)} · form ${Number(p.form || 0).toFixed(1)}`,
        tone: "up",
        barPct: (Number(p.ep || 0) / Math.max(1, Number((t.captain_picks || [])[0]?.ep) || 1)) * 100,
      })
    ).join("");

    const diffs = (t.differentials || []).map((p, i) =>
      playerRow(p, i, {
        valueHtml: `${Number(p.own || 0).toFixed(1)}% · form ${Number(p.form || 0).toFixed(1)}`,
        tone: "hot",
        barPct: (Number(p.form || 0) / Math.max(1, Number((t.differentials || [])[0]?.form) || 1)) * 100,
      })
    ).join("");

    const xgi = (t.xgi_leaders || []).map((p, i) =>
      playerRow(p, i, {
        valueHtml: `xGI ${Number(p.xgi || 0).toFixed(2)}`,
        tone: "up",
        barPct: (Number(p.xgi || 0) / Math.max(0.01, Number((t.xgi_leaders || [])[0]?.xgi) || 0.01)) * 100,
      })
    ).join("");

    const doubts = (t.doubts || []).map((p) => `
      <div class="alert ${p.owned ? "is-owned" : ""}">
        <span class="nm">${esc(p.name)}</span>
        <span class="meta">${esc(p.team || "")} · ${esc(p.pos || "")} · ${Number(p.own || 0).toFixed(1)}%</span>
        ${p.status && p.status !== "a" ? `<span class="flag status">${esc(String(p.status).toUpperCase())}</span>` : ""}
        ${p.chance != null ? `<span class="flag doubt">${esc(p.chance)}%</span>` : ""}
        ${p.owned ? `<span class="flag easy">you</span>` : ""}
        ${p.news ? `<span class="meta">${esc(p.news)}</span>` : ""}
      </div>`).join("");

    const byPos = t.by_pos || {};
    const posHtml = ["GK", "DEF", "MID", "FWD"].map((pos) => {
      const list = byPos[pos] || [];
      const rows = list.map((p, i) =>
        playerRow(p, i, {
          valueHtml: `EP ${Number(p.ep || 0).toFixed(1)}`,
          tone: "up",
          barPct: (Number(p.ep || 0) / Math.max(1, Number(list[0]?.ep) || 1)) * 100,
        })
      ).join("");
      return trendCard(`${pos} EP`, "Highest expected points next GW", rows, "neutral");
    }).join("");

    const bd = (hub.fixtures && hub.fixtures.blank_double) || [];
    const bdHtml = bd.some((g) => (g.blanks || []).length || (g.doubles || []).length)
      ? `<div class="bd-list">${bd.map((g) => {
          if (!(g.blanks || []).length && !(g.doubles || []).length) {
            return `<div class="bd-row muted">GW${esc(g.gameweek)} · all clubs play once</div>`;
          }
          return `<div class="bd-row">
            <strong>GW${esc(g.gameweek)}</strong>
            ${(g.doubles || []).length ? `<span class="flag easy">DGW ${(g.doubles || []).map(esc).join(", ")}</span>` : ""}
            ${(g.blanks || []).length ? `<span class="flag hard">BGW ${(g.blanks || []).map(esc).join(", ")}</span>` : ""}
          </div>`;
        }).join("")}</div>`
      : `<p class="empty">No blanks or doubles in the next few gameweeks.</p>`;

    el.innerHTML = `
      <p class="section-label">Your squad radar</p>
      ${alertHtml}

      <p class="section-label" style="margin-top:1.1rem;">Your squad fixtures (next 5)</p>
      ${squadFxHtml}

      <p class="section-label" style="margin-top:1.1rem;">Picks &amp; underlying</p>
      <div class="trend-grid">
        ${trendCard("Captain candidates", "EP + form + fixture ease (fit players)", captains, "up")}
        ${trendCard("Differentials", "Under 8% owned with strong form", diffs, "hot")}
        ${trendCard("xGI leaders", "Expected goal involvement this season", xgi, "neutral")}
        <article class="trend-card tone-down">
          <h3>Availability watch</h3>
          <p class="hint">Flags, % chance, and news — owned players first</p>
          <div class="alert-list">${doubts || `<p class="empty">No notable doubts.</p>`}</div>
        </article>
      </div>

      <p class="section-label" style="margin-top:1.1rem;">Best EP by position</p>
      <div class="trend-grid">${posHtml}</div>

      <p class="section-label" style="margin-top:1.1rem;">Market &amp; form this gameweek</p>
      <div class="trend-grid">
        ${trendCard("Price risers", "Overnight / GW price climbs", priceUp, "up")}
        ${trendCard("Price fallers", "Players dropping in value", priceDown, "down")}
        ${trendCard("Transfer magnets", "Net transfers in this GW", transfersIn, "up")}
        ${trendCard("Being dumped", "Net transfers out this GW", transfersOut, "down")}
        ${trendCard("Form heating up", "Form well above season PPG", formHot, "hot")}
        ${trendCard("Form cooling", "Form below season PPG", formCold, "cold")}
      </div>

      <p class="section-label">Fixture runs (next 5)</p>
      <div class="run-grid">
        ${runBlock("Easiest schedules", "Higher ease = softer opponents", fdr.easiest, "up")}
        ${runBlock("Toughest schedules", "Lower ease = harder stretch", fdr.hardest, "down")}
      </div>

      <p class="section-label" style="margin-top:1.1rem;">Blanks &amp; doubles</p>
      ${bdHtml}
    `;
  }

  function renderFixtures(hub) {
    const fxEl = $("hubFixtures");
    if (!fxEl) return;
    const byGw = (hub.fixtures && hub.fixtures.by_gw) || [];
    const bd = (hub.fixtures && hub.fixtures.blank_double) || [];
    if (!byGw.length) {
      fxEl.innerHTML = `<p class="empty">No fixtures in the next window.</p>`;
      return;
    }
    const bdNote = bd.slice(0, 3).map((g) => {
      if (!(g.blanks || []).length && !(g.doubles || []).length) return "";
      return `<div class="bd-row">
        <strong>GW${esc(g.gameweek)}</strong>
        ${(g.doubles || []).length ? `<span class="flag easy">DGW ${(g.doubles || []).map(esc).join(", ")}</span>` : ""}
        ${(g.blanks || []).length ? `<span class="flag hard">BGW ${(g.blanks || []).map(esc).join(", ")}</span>` : ""}
      </div>`;
    }).filter(Boolean).join("");

    fxEl.innerHTML = `
      ${bdNote ? `<div class="bd-list" style="margin-bottom:1rem;">${bdNote}</div>` : ""}
      ${byGw.map((block) => {
        const matches = block.matches || [];
        const rows = matches.length
          ? matches.map((m) => {
              const sh = m.score && m.score.h != null ? m.score.h : null;
              const sa = m.score && m.score.a != null ? m.score.a : null;
              const mid = (sh != null && sa != null)
                ? `<span class="score-mid">${esc(sh)}–${esc(sa)}</span>`
                : `<span class="score-mid vs">vs</span>`;
              const hf = m.home && m.home.fdr != null ? Number(m.home.fdr) : 3;
              const af = m.away && m.away.fdr != null ? Number(m.away.fdr) : 3;
              return `<div class="fx-row">
                <span class="kick">${esc(fmtKick(m.kickoff))}</span>
                <div class="side home">
                  <span class="tname">${esc((m.home && (m.home.short || m.home.name)) || "—")}</span>
                  <span class="fdr-dot fdr-${hf}">${hf}</span>
                </div>
                ${mid}
                <div class="side away">
                  <span class="fdr-dot fdr-${af}">${af}</span>
                  <span class="tname">${esc((m.away && (m.away.short || m.away.name)) || "—")}</span>
                </div>
              </div>`;
            }).join("")
          : `<p class="empty">No matches listed.</p>`;
        return `<div class="hub-gw-block"><h3>Gameweek ${esc(block.gameweek)}</h3><div class="fx-list">${rows}</div></div>`;
      }).join("")}`;
  }

  function findRunBlocks(cells, { minLen = 2 } = {}) {
    const blocks = [];
    let i = 0;
    const list = cells || [];
    while (i < list.length) {
      const c = list[i];
      if (!c || c.blank || c.fdr == null) {
        i += 1;
        continue;
      }
      const fdr = Number(c.fdr);
      let kind = null;
      if (fdr <= 2) kind = "easy";
      else if (fdr >= 4) kind = "hard";
      else {
        i += 1;
        continue;
      }
      let j = i + 1;
      let sum = fdr;
      while (j < list.length) {
        const n = list[j];
        if (!n || n.blank || n.fdr == null) break;
        const nf = Number(n.fdr);
        if (kind === "easy" && nf > 2) break;
        if (kind === "hard" && nf < 4) break;
        sum += nf;
        j += 1;
      }
      const len = j - i;
      if (len >= minLen) {
        blocks.push({
          kind,
          start: i,
          end: j - 1,
          len,
          avg: sum / len,
        });
      }
      i = j;
    }
    return blocks;
  }

  function runAt(blocks, index) {
    return (blocks || []).find((b) => index >= b.start && index <= b.end) || null;
  }

  function runEdgeClass(i, run) {
    if (!run || i < run.start || i > run.end) return "";
    if (run.start === run.end) return "run-start run-end";
    if (i === run.start) return "run-start";
    if (i === run.end) return "run-end";
    return "run-mid";
  }

  function runSummary(blocks) {
    if (!blocks || !blocks.length) return "";
    return blocks.map((b) => {
      const label = b.kind === "easy" ? "easy" : "hard";
      return `${b.len}${label[0]}`;
    }).join(" · ");
  }

  function renderFdr(hub) {
    const fdrEl = $("hubFdr");
    if (!fdrEl) return;
    const fdr = hub.fdr || {};
    const gws = fdr.gws || hub.horizon_gws || [];
    const rows = fdr.rows || [];
    const easiestIds = new Set((fdr.easiest || []).slice(0, 4).map((r) => r.id));
    const hardestIds = new Set((fdr.hardest || []).slice(0, 4).map((r) => r.id));
    const legend = [1, 2, 3, 4, 5].map((n) =>
      `<span><span class="fdr-dot fdr-${n}">${n}</span> ${n === 1 ? "easiest" : n === 5 ? "hardest" : ""}</span>`
    ).join("");
    if (!rows.length) {
      fdrEl.innerHTML = `<p class="empty">FDR matrix unavailable.</p>`;
      return;
    }
    const head = `<tr>
      <th class="team">Team</th>
      ${gws.map((g) => `<th>GW${esc(g)}</th>`).join("")}
      <th>Ease</th>
    </tr>`;
    const body = rows.map((r) => {
      const cells = r.cells || [];
      const blocks = findRunBlocks(cells, { minLen: 2 });
      const rowClass = [
        easiestIds.has(r.id) ? "is-easiest" : "",
        hardestIds.has(r.id) ? "is-hardest" : "",
      ].filter(Boolean).join(" ");
      const summary = runSummary(blocks);
      const badges = [
        easiestIds.has(r.id) ? `<span class="team-badge easy">soft</span>` : "",
        hardestIds.has(r.id) ? `<span class="team-badge hard">tough</span>` : "",
        summary ? `<span class="run-summary" title="Contiguous FDR blocks across the horizon">${esc(summary)}</span>` : "",
      ].join("");
      const cellHtml = cells.map((c, i) => {
        const block = runAt(blocks, i);
        if (c.blank || c.fdr == null) {
          return `<td><span class="fdr-cell blank">—</span></td>`;
        }
        const f = Number(c.fdr);
        const runCls = [];
        if (block) {
          runCls.push(block.kind === "easy" ? "run-easy" : "run-hard", runEdgeClass(i, block));
        }
        const title = [
          c.label || "",
          block
            ? ` · ${block.kind} run of ${block.len} (avg FDR ${block.avg.toFixed(1)})`
            : (f === 3 ? " · middling" : ""),
        ].join("");
        return `<td class="${block ? `in-run in-run-${block.kind}` : ""}"><span class="fdr-cell fdr-${f} ${runCls.join(" ")}" title="${esc(title)}"><span class="opp">${esc(c.opp || "")}</span><span class="ha">${c.home ? "H" : "A"}</span></span></td>`;
      }).join("");
      return `<tr class="${rowClass}">
        <td class="team">${esc(r.short || r.name || "")}${badges}</td>
        ${cellHtml}
        <td class="ease">${r.run_ease != null ? esc(r.run_ease) : "—"}</td>
      </tr>`;
    }).join("");
    fdrEl.innerHTML = `
      <div class="fdr-legend">${legend}<span>Sorted by run ease · green = easier</span></div>
      <div class="fdr-run-key">
        <span class="swatch"><span class="box easy"></span> Easy block (FDR 1–2, ≥2 GWs)</span>
        <span class="swatch"><span class="box hard"></span> Hard block (FDR 4–5, ≥2 GWs)</span>
        <span class="swatch">FDR 3 stays unblocked · blanks break a run</span>
        <span class="swatch"><span class="run-summary">2h · 4e</span> Per-team block summary</span>
      </div>
      <div class="fdr-wrap"><table class="fdr-table fdr-table--blocks"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
  }

  function renderStats(hub) {
    const statsEl = $("hubStats");
    if (!statsEl) return;
    const allPlayers = (hub.stats && hub.stats.players) || [];

    function paintStats() {
      const pos = hubStatsFilter.pos || "ALL";
      const q = (hubStatsFilter.q || "").trim().toLowerCase();
      let list = allPlayers.slice();
      if (pos !== "ALL") list = list.filter((p) => p.pos === pos);
      if (hubStatsFilter.ownedOnly) list = list.filter((p) => p.owned);
      if (q) {
        list = list.filter((p) =>
          String(p.name || "").toLowerCase().includes(q)
          || String(p.team || "").toLowerCase().includes(q)
        );
      }
      const key = hubStatsSort.key;
      const dir = hubStatsSort.dir;
      list.sort((a, b) => {
        const av = a[key];
        const bv = b[key];
        if (typeof av === "string" || typeof bv === "string") {
          return dir * String(av || "").localeCompare(String(bv || ""));
        }
        return dir * ((Number(av) || 0) - (Number(bv) || 0));
      });
      list = list.slice(0, 100);

      const cols = [
        { key: "name", label: "Player", left: true },
        { key: "pos", label: "Pos" },
        { key: "price", label: "£" },
        { key: "price_delta", label: "£Δ" },
        { key: "form", label: "Form" },
        { key: "delta_form", label: "FormΔ" },
        { key: "ppg", label: "PPG" },
        { key: "ep", label: "EP" },
        { key: "xgi", label: "xGI" },
        { key: "net_transfers", label: "Net T" },
        { key: "own", label: "Own%" },
        { key: "fixture_ease", label: "Fix" },
        { key: "pts", label: "Pts" },
      ];
      const th = cols.map((c) => {
        const on = hubStatsSort.key === c.key;
        const arrow = on ? (hubStatsSort.dir < 0 ? " ↓" : " ↑") : "";
        return `<th class="${c.left ? "left" : ""}${on ? " is-sort" : ""}" data-sort="${c.key}">${esc(c.label)}${arrow}</th>`;
      }).join("");

      const body = list.map((p) => {
        const dForm = Number(p.delta_form || 0);
        const dPrice = Number(p.price_delta || 0);
        const net = Number(p.net_transfers || 0);
        return `<tr class="${p.owned ? "is-owned" : ""}">
          <td class="left"><span class="nm">${esc(p.name)}</span><span class="meta">${esc(p.team || "")}</span></td>
          <td>${esc(p.pos)}</td>
          <td>${Number(p.price || 0).toFixed(1)}</td>
          <td class="${dPrice > 0 ? "delta-up" : dPrice < 0 ? "delta-down" : ""}">${esc(fmtPriceDelta(dPrice))}</td>
          <td>${Number(p.form || 0).toFixed(1)}</td>
          <td class="${dForm > 0 ? "delta-up" : dForm < 0 ? "delta-down" : ""}">${dForm > 0 ? "+" : ""}${dForm.toFixed(1)}</td>
          <td>${Number(p.ppg || 0).toFixed(1)}</td>
          <td>${Number(p.ep || 0).toFixed(1)}</td>
          <td>${Number(p.xgi || 0).toFixed(2)}</td>
          <td class="${net > 0 ? "delta-up" : net < 0 ? "delta-down" : ""}">${esc(fmtNet(net))}</td>
          <td>${Number(p.own || 0).toFixed(1)}</td>
          <td>${p.fixture_ease != null ? Number(p.fixture_ease).toFixed(1) : "—"}</td>
          <td>${esc(p.pts ?? "—")}</td>
        </tr>`;
      }).join("");

      statsEl.innerHTML = `
        <div class="hub-stats-toolbar">
          <select id="hubPosFilter" aria-label="Position filter">
            ${["ALL", "GK", "DEF", "MID", "FWD"].map((p) =>
              `<option value="${p}" ${pos === p ? "selected" : ""}>${p === "ALL" ? "All positions" : p}</option>`
            ).join("")}
          </select>
          <input id="hubSearch" type="search" placeholder="Search player or club" value="${esc(hubStatsFilter.q || "")}" />
          <label class="owned-toggle">
            <input type="checkbox" id="hubOwnedOnly" ${hubStatsFilter.ownedOnly ? "checked" : ""} /> Owned
          </label>
        </div>
        <div class="hub-stats-wrap">
          <table class="hub-stats-table">
            <thead><tr>${th}</tr></thead>
            <tbody>${body || `<tr><td colspan="13" class="left empty">No players match.</td></tr>`}</tbody>
          </table>
        </div>`;

      $("hubPosFilter")?.addEventListener("change", (ev) => {
        hubStatsFilter.pos = ev.target.value;
        paintStats();
      });
      const search = $("hubSearch");
      if (search) {
        let timer;
        search.addEventListener("input", () => {
          clearTimeout(timer);
          timer = setTimeout(() => {
            hubStatsFilter.q = search.value;
            paintStats();
          }, 160);
        });
      }
      $("hubOwnedOnly")?.addEventListener("change", (ev) => {
        hubStatsFilter.ownedOnly = !!ev.target.checked;
        paintStats();
      });
      statsEl.querySelectorAll("th[data-sort]").forEach((thEl) => {
        thEl.addEventListener("click", () => {
          const k = thEl.getAttribute("data-sort");
          if (hubStatsSort.key === k) hubStatsSort.dir *= -1;
          else {
            hubStatsSort.key = k;
            hubStatsSort.dir = (k === "name" || k === "pos") ? 1 : -1;
          }
          paintStats();
        });
      });
    }
    paintStats();
  }

  function renderHub(hub) {
    const panes = ["hubOverview", "hubFixtures", "hubFdr", "hubStats"];
    if (!hub || !hub.ok) {
      const msg = `<p class="empty">${esc((hub && hub.reason) || "Hub data unavailable")}</p>`;
      panes.forEach((id) => { const el = $(id); if (el) el.innerHTML = msg; });
      return;
    }
    renderOverview(hub);
    renderFixtures(hub);
    renderFdr(hub);
    renderStats(hub);
  }

  function renderChrome(data) {
    const m = data.manager || {};
    const d = data.deadline || {};
    const hub = data.hub || {};
    const lede = $("lede");
    if (lede) {
      lede.textContent =
        `${m.team_name || "Your team"} · trends, fixtures & FDR for GW${hub.current_gw ?? d.gameweek ?? "—"}`;
    }
    const snap = $("snap");
    if (snap) {
      const alerts = ((hub.trends || {}).owned_alerts || []).length;
      snap.innerHTML = [
        `<div class="snap-pill"><div class="k">Gameweek</div><div class="v">${esc(hub.current_gw ?? d.gameweek ?? "—")}</div></div>`,
        `<div class="snap-pill"><div class="k">Deadline</div><div class="v">${esc(fmtHours(d.hours_to_deadline))}</div></div>`,
        `<div class="snap-pill"><div class="k">Squad flags</div><div class="v">${alerts}</div></div>`,
      ].join("");
    }
    const foot = $("foot");
    if (foot) {
      foot.textContent = `Updated ${data.generated_at ? new Date(data.generated_at).toLocaleString() : "—"} · refreshes every 45s`;
    }
  }

  async function tick() {
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      const data = await res.json();
      renderChrome(data);
      renderHub(data.hub);
    } catch (err) {
      const lede = $("lede");
      if (lede) lede.textContent = `Could not load hub data (${err}).`;
    }
  }

  initHubTabs();
  tick();
  setInterval(tick, 45000);
})();
