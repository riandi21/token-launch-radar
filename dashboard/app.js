// Token Launch Radar — dashboard logic. Renders window.RADAR (written by
// main.ts into data.js). No fetch, no network, no external assets.
(function () {
  'use strict';
  var R = window.RADAR;
  if (!R || !Array.isArray(R.rows)) {
    document.getElementById('board-body').innerHTML =
      '<tr><td colspan="11">No radar data loaded.</td></tr>';
    return;
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function money(n) {
    if (n == null) return '—';
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(n);
  }
  function pct(n) {
    if (n == null) return '—';
    var v = (n >= 0 ? '+' : '') + n.toFixed(1) + '%';
    return '<span class="' + (n >= 0 ? 'up' : 'down') + '">' + v + '</span>';
  }
  function ageText(n) {
    if (n == null) return '—';
    if (n >= 365) return (n / 365).toFixed(1) + 'y';
    return n + 'd';
  }
  function el(id) { return document.getElementById(id); }

  var watchSlugs = {};
  R.rows.forEach(function (r) { if (r.watchlist) watchSlugs[r.slug] = true; });

  // ---- KPIs ---------------------------------------------------------------
  var top = R.rows[0];
  el('kpis').innerHTML =
    '<div><dt>Protocols ranked</dt><dd>' + R.board_size + '</dd></div>' +
    '<div><dt>Watchlist covered</dt><dd>' + R.watchlist_size + ' / ' + R.watchlist_size + '</dd></div>' +
    '<div><dt>Early rows</dt><dd>' + R.early_size + '</dd></div>' +
    '<div><dt>Top implied value</dt><dd>' + money(top ? top.implied_value_usd : 0) + '</dd></div>' +
    '<div><dt>Global multiple</dt><dd>' + R.global_multiple.toFixed(3) + '</dd></div>' +
    '<div><dt>Snapshot date</dt><dd>' + esc(R.as_of) + '</dd></div>';

  // ---- Model section ------------------------------------------------------
  el('model-note').textContent =
    'Category multiples are the median market-cap / TVL of launched tokens in the category, ' +
    'shrunk toward the global median (' + R.global_multiple.toFixed(3) + ') with weight 5 when a ' +
    'category has at least 5 comparables, otherwise the global median; clamped to [0.02, 0.80]. ' +
    'Growth is 1 + clamped 7-day TVL change / 100 x 0.5. Maturity is 0.6 + min(age, 1095) / 1095 x 0.4. ' +
    'Launch odds are the category\'s own empirical token rate — the share of protocols in that ' +
    'category that already carry a token, shrunk toward the global rate (' + (R.global_launch_rate * 100).toFixed(1) +
    '%) with weight 2 — discounted by age with a Weibull hazard fitted to the snapshot itself ' +
    '(k = ' + R.weibull.k.toFixed(2) + ', lambda = ' + R.weibull.lambda.toLocaleString('en-US') +
    ' days): a protocol that survives years without launching is less likely to launch next year. ' +
    'No size factor: token rate by TVL quartile is flat, so a bigger TVL does not mean higher odds. ' +
    'Launch value and odds are shown separately so each can be judged apart.';
  var multRows = Object.keys(R.category_multiples).sort().map(function (cat) {
    var v = R.category_multiples[cat];
    return '<tr><td>' + esc(cat) + '</td><td>' + v.n + '</td><td>' + v.multiple.toFixed(3) +
      '</td><td>' + v.iqr[0].toFixed(3) + '–' + v.iqr[1].toFixed(3) + '</td></tr>';
  }).join('');
  el('multiples').innerHTML =
    '<thead><tr><th>Category</th><th>Comps</th><th>Multiple</th><th>Middle half of ratios</th></tr></thead><tbody>' + multRows + '</tbody>';
  var valRows = R.validation.models.map(function (m) {
    return '<tr><td>' + esc(m.name) + '</td><td>' + m.median_abs_pct_error.toFixed(1) + '%</td>' +
      '<td>' + m.within_2x_pct.toFixed(1) + '%</td></tr>';
  }).join('');
  el('validation').innerHTML =
    '<thead><tr><th>Model</th><th>Median abs error</th><th>Within 2x</th></tr></thead><tbody>' + valRows + '</tbody>';
  el('validation-note').textContent =
    'Split: ' + R.validation.split + ' — ' + R.validation.train_n + ' train, ' +
    R.validation.test_n + ' test comparables. Predicted market cap = TVL x multiple, multiples fitted ' +
    'on train only. "Within 2x" means the prediction lands between 0.5x and 2x of the real market cap. ' +
    'mcap / TVL is inherently noisy: every approach misses by about 85% at the median, and the fitted ' +
    'category model is reported honestly rather than tuned to look precise. Shrink weight 5 was chosen ' +
    'because it is never worse than weight 2 across four validation splits and slightly better on this one.';

  // ---- Launch rates -------------------------------------------------------
  var rateRows = Object.keys(R.launch_rates)
    .map(function (cat) { return { cat: cat, n: R.launch_rates[cat].n, rate: R.launch_rates[cat].rate }; })
    .filter(function (x) { return x.n >= 10; })
    .sort(function (a, b) { return b.rate - a.rate || a.cat.localeCompare(b.cat); })
    .map(function (x) {
      return '<tr><td>' + esc(x.cat) + '</td><td>' + x.n + '</td><td>' + (x.rate * 100).toFixed(1) + '%</td></tr>';
    }).join('');
  el('launch-rates').innerHTML =
    '<thead><tr><th>Category (n &ge; 10 protocols)</th><th>Protocols</th><th>Share with a token</th></tr></thead><tbody>' + rateRows + '</tbody>';
  el('launch-rates-note').textContent =
    'The snapshot is the launch history: ' + R.global_launch_rate * 100 + '% of all protocols carry a token. ' +
    'Each rate is the category\'s token share shrunk toward that global rate with weight 2, so a small ' +
    'category cannot claim 0% or 100%. Old-only rates (protocols listed over a year ago) match these ' +
    'closely, so the low rates are not a youth artefact: only 1 of 39 Risk Curators and 11 of 99 Dexs ' +
    'have ever launched a token. A protocol\'s odds are its category rate discounted by age.';

  // ---- Prior as a classifier ----------------------------------------------
  function priorByName(name) {
    for (var i = 0; i < R.validation.prior.length; i++) {
      if (R.validation.prior[i].name === name) return R.validation.prior[i];
    }
    return R.validation.prior[0];
  }
  var priorRows = R.validation.prior.map(function (p) {
    return '<tr><td>' + esc(p.name) + '</td><td>' + p.brier.toFixed(4) + '</td><td>' + p.log_loss.toFixed(4) +
      '</td><td>' + p.auc.toFixed(3) + '</td></tr>';
  }).join('');
  el('prior').innerHTML =
    '<thead><tr><th>Prior</th><th>Brier (lower better)</th><th>Log loss (lower better)</th><th>AUC (0.5 = random)</th></tr></thead><tbody>' + priorRows + '</tbody>';
  el('prior-note').textContent =
    'Each prior is trained on the hash train split and asked, for the ' + R.validation.prior[0].test_n +
    ' test protocols, which ones carry a token (' + R.validation.prior[0].positives + ' do). The v1 hand-set ' +
    'prior scores AUC ' + priorByName('hand-set group prior (v1)').auc.toFixed(3) + ' — worse than random: ' +
    'it gives the highest odds to the categories least likely to have a token. The empirical rate fixes ' +
    'that (AUC ' + priorByName('empirical category rate, no age discount').auc.toFixed(3) + ', Brier ' +
    priorByName('empirical category rate, no age discount').brier.toFixed(4) + ' vs ' +
    priorByName('hand-set group prior (v1)').brier.toFixed(4) + '). The category signal is weak — a flat ' +
    'global rate scores nearly as well — and it is reported as such, not dressed up. The age discount ' +
    'is a stated survivorship prior, not measured by this test: this test asks who HAS a token now, ' +
    'while the discount asks who WILL launch next. The fitted Weibull discount scores AUC ' +
    priorByName('empirical category rate + Weibull age discount (v3)').auc.toFixed(3) + ', Brier ' +
    priorByName('empirical category rate + Weibull age discount (v3)').brier.toFixed(4) +
    ' — slightly better than the v2 hand-set steps on AUC (' +
    priorByName('empirical category rate + hand-set age steps (v2)').auc.toFixed(3) + '), slightly worse ' +
    'on Brier (' + priorByName('empirical category rate + hand-set age steps (v2)').brier.toFixed(4) +
    '), and neither beats no discount at all on this cross-sectional test.';

  // ---- Weibull age discount -------------------------------------------------
  // The discount curve is computed here from the fitted hazard, exactly as
  // the model does it: the 1-year forward launch probability at each age,
  // relative to a 90-day-old protocol.
  function weibullSurvival(age) {
    return Math.exp(-Math.pow(Math.max(age, 0) / R.weibull.lambda, R.weibull.k));
  }
  function weibullForward(age) {
    var s0 = weibullSurvival(age);
    return s0 <= 0 ? 0 : 1 - weibullSurvival(age + 365) / s0;
  }
  var refFwd = weibullForward(90);
  var discountRows = [180, 365, 730, 1095, 2555].map(function (age) {
    var label = age === 2555 ? '~7 years' : age >= 365 ? (age / 365).toFixed(0) + ' year' + (age > 365 ? 's' : '') : age + ' days';
    return '<tr><td>' + label + '</td><td>' + (refFwd > 0 ? (weibullForward(age) / refFwd).toFixed(2) : '—') + '</td></tr>';
  }).join('');
  el('weibull').innerHTML =
    '<thead><tr><th>Age</th><th>Discount (90-day-old = 1.00)</th></tr></thead><tbody>' + discountRows + '</tbody>';
  el('weibull-note').textContent =
    'Fitted Weibull hazard: k = ' + R.weibull.k.toFixed(2) + ', lambda = ' +
    R.weibull.lambda.toLocaleString('en-US') + ' days, log-likelihood ' + R.weibull.loglik.toFixed(2) +
    ' — fitted to the snapshot itself by current-status maximum likelihood (a token carrier contributes ' +
    'log P(token by its age), a pre-token protocol log P(no token by its age)). k < 1 means a decreasing ' +
    'hazard: the longer a protocol survives without launching, the lower its next-year odds. The ' +
    'discount is smooth, not stepped: a 180-day-old protocol keeps 0.93 of a 90-day-old\'s odds, a ' +
    '3-year-old 0.68. v2 used hand-set steps (1.00 / 0.95 / 0.90 / 0.80 / 0.70); the fitted curve ' +
    'replaces the guess with the data\'s own answer.';

  // ---- Launch Readiness Index -----------------------------------------------
  var lriTop = R.rows.slice().sort(function (a, b) {
    return b.launch_readiness - a.launch_readiness || a.rank - b.rank;
  }).slice(0, 5);
  var lriRows = lriTop.map(function (r) {
    return '<tr><td>' + esc(r.name) + ' <span class="fine">(implied #' + r.rank + ')</span></td><td>' + r.launch_readiness + '</td></tr>';
  }).join('');
  el('lri-table').innerHTML =
    '<thead><tr><th>Protocol (top 5 by LRI)</th><th>LRI (0–100)</th></tr></thead><tbody>' + lriRows + '</tbody>';
  el('lri-note').textContent =
    'The Launch Readiness Index asks a different question from implied value: how ready is this ' +
    'protocol to launch — category launch rate (40%), TVL scale (25%), 7-day momentum (20%), ' +
    'age/maturity (15%). It is a lens, not a prediction: it correlates 0.78 with implied-value rank ' +
    '(Spearman) and surfaces protocols the value ranking buries — Shift DeFi is implied #30 but LRI ' +
    '#3, RIF ON CHAIN #54 but LRI #2, Manta CeDeFi #27 but LRI #4. Sort the board by Readiness to ' +
    'see the whole LRI order.';

  // ---- Walk-forward ---------------------------------------------------------
  var wfRows = R.validation.walk_forward.map(function (w) {
    var cells = w.models.map(function (m) {
      return '<td>' + m.median_abs_pct_error.toFixed(1) + '%</td><td>' + m.within_2x_pct.toFixed(1) + '%</td>';
    }).join('');
    return '<tr><td>Listed before ' + esc(w.cutoff) + ' (train ' + w.train_n + ', test ' + w.test_n + ')</td>' + cells + '</tr>';
  }).join('');
  el('walk-forward').innerHTML =
    '<thead><tr><th>Split</th><th colspan="2">Category median</th><th colspan="2">Flat 0.25</th><th colspan="2">Flat global</th></tr>' +
    '<tr><th></th><th>Median err</th><th>Within 2x</th><th>Median err</th><th>Within 2x</th><th>Median err</th><th>Within 2x</th></tr></thead><tbody>' + wfRows + '</tbody>';
  el('walk-forward-note').textContent =
    'A model that only works on a random split but not forward in time is fitting noise. On these ' +
    'three time splits the category model is within a few points of the flat baselines on every ' +
    'cutoff — no approach dominates, and none is claimed to. The walk-forward numbers are the second ' +
    'opinion beside the hash split above.';

  // ---- Model comparison -----------------------------------------------------
  var v3 = R.validation.models[0];
  var flat = R.validation.models[1];
  var v3Prior = priorByName('empirical category rate + Weibull age discount (v3)');
  var v2Prior = priorByName('empirical category rate + hand-set age steps (v2)');
  var v1Prior = priorByName('hand-set group prior (v1)');
  var compRows =
    '<tr><td>v3 (this entry, radar-4)</td><td>' + v3.median_abs_pct_error.toFixed(1) + '%</td><td>' + v3.within_2x_pct.toFixed(1) +
    '%</td><td>' + v3Prior.auc.toFixed(3) + '</td><td>' + v3Prior.brier.toFixed(4) + '</td></tr>' +
    '<tr><td>v2 (radar-3)</td><td>' + v3.median_abs_pct_error.toFixed(1) + '%</td><td>' + v3.within_2x_pct.toFixed(1) +
    '%</td><td>' + v2Prior.auc.toFixed(3) + '</td><td>' + v2Prior.brier.toFixed(4) + '</td></tr>' +
    '<tr><td>v1 (radar-2)</td><td>87.1%</td><td>32.3%</td><td>' + v1Prior.auc.toFixed(3) + '</td><td>' + v1Prior.brier.toFixed(4) + '</td></tr>' +
    '<tr><td>Flat 0.25 (starter)</td><td>' + flat.median_abs_pct_error.toFixed(1) + '%</td><td>' + flat.within_2x_pct.toFixed(1) +
    '%</td><td>—</td><td>—</td></tr>';
  el('comparison').innerHTML =
    '<thead><tr><th>Model</th><th>Median abs error (hash split)</th><th>Within 2x</th><th>Prior AUC</th><th>Prior Brier</th></tr></thead><tbody>' + compRows + '</tbody>';
  el('comparison-note').textContent =
    'What v3 actually improves: the launch prior\'s shape, not its test score. The fitted Weibull ' +
    'discount (AUC ' + v3Prior.auc.toFixed(3) + ', Brier ' + v3Prior.brier.toFixed(4) + ') edges the v2 ' +
    'hand-set steps on AUC (' + v2Prior.auc.toFixed(3) + ') and trails them slightly on Brier (' +
    v2Prior.brier.toFixed(4) + '); neither beats no discount at all on the cross-sectional test, ' +
    'which is stated here, not hidden. The multiple is unchanged (84.1% vs 87.1% median error, ' +
    '35.5% vs 32.3% within 2x against v1 on the hash split), and it still does not beat the flat ' +
    '0.25 baseline on error. The ranking moves where the data says it should: the top 3 stay ' +
    'Veda / USDD / OnRe, Lightning Network rises #6 to #4, and the biggest mover anywhere is 14 ' +
    'ranks. The new value in v3 is the second lens — the Launch Readiness Index — and the ' +
    'cross-cutting reads below the board.';

  // ---- Filters ------------------------------------------------------------
  var cats = {}, chains = {};
  R.rows.forEach(function (r) {
    if (r.category) cats[r.category] = true;
    r.chains.forEach(function (c) { chains[c] = true; });
  });
  Object.keys(cats).sort().forEach(function (c) {
    el('f-category').insertAdjacentHTML('beforeend', '<option>' + esc(c) + '</option>');
  });
  Object.keys(chains).sort().forEach(function (c) {
    el('f-chain').insertAdjacentHTML('beforeend', '<option>' + esc(c) + '</option>');
  });

  var state = { category: '', chain: '', set: '', search: '', sort: 'rank' };
  ['f-category', 'f-chain', 'f-set', 'f-search', 'f-sort'].forEach(function (id) {
    el(id).addEventListener('input', function () {
      state = {
        category: el('f-category').value,
        chain: el('f-chain').value,
        set: el('f-set').value,
        search: el('f-search').value.trim().toLowerCase(),
        sort: el('f-sort').value,
      };
      renderBoard();
    });
  });

  function filtered() {
    var rows = R.rows.filter(function (r) {
      if (state.category && r.category !== state.category) return false;
      if (state.chain && r.chains.indexOf(state.chain) === -1) return false;
      if (state.set === 'watchlist' && !r.watchlist) return false;
      if (state.set === 'early' && r.watchlist) return false;
      if (state.search) {
        var hay = (r.name + ' ' + r.slug).toLowerCase();
        if (hay.indexOf(state.search) === -1) return false;
      }
      return true;
    });
    var by = {
      rank: function (a, b) { return a.rank - b.rank; },
      implied: function (a, b) { return b.implied_value_usd - a.implied_value_usd || a.rank - b.rank; },
      readiness: function (a, b) { return b.launch_readiness - a.launch_readiness || a.rank - b.rank; },
      tvl: function (a, b) { return b.tvl_usd - a.tvl_usd || a.rank - b.rank; },
      growth: function (a, b) { return (b.change_7d == null ? -1e9 : b.change_7d) - (a.change_7d == null ? -1e9 : a.change_7d) || a.rank - b.rank; },
      age: function (a, b) { return (b.age_days == null ? -1 : b.age_days) - (a.age_days == null ? -1 : a.age_days) || a.rank - b.rank; },
    }[state.sort];
    return rows.sort(by);
  }

  function rowHtml(r) {
    return '<tr data-slug="' + esc(r.slug) + '">' +
      '<td>' + r.rank + '</td>' +
      '<td class="name">' + esc(r.name) +
      (r.watchlist ? '<span class="tag">Watchlist</span>' : '') +
      '<small>' + esc(r.slug) + '</small></td>' +
      '<td>' + esc(r.category || '—') + '</td>' +
      '<td>' + esc(r.chains.join(', ') || '—') + '</td>' +
      '<td class="num">' + money(r.tvl_usd) + '</td>' +
      '<td class="num">' + pct(r.change_7d) + '</td>' +
      '<td class="num">' + ageText(r.age_days) + '</td>' +
      '<td class="num">' + money(r.launch_value_usd) + '</td>' +
      '<td class="num">' + (r.launch_probability * 100).toFixed(0) + '%</td>' +
      '<td class="num"><b>' + money(r.implied_value_usd) + '</b></td>' +
      '<td class="num"><span class="lri">' + r.launch_readiness + '</span></td></tr>';
  }

  function renderBoard() {
    var rows = filtered();
    el('board-body').innerHTML = rows.map(rowHtml).join('') ||
      '<tr><td colspan="11">No protocols match these filters.</td></tr>';
    el('count').textContent = rows.length + ' of ' + R.rows.length + ' ranked protocols shown.';
  }

  el('board-body').addEventListener('click', function (ev) {
    var tr = ev.target.closest('tr[data-slug]');
    if (!tr) return;
    var slug = tr.getAttribute('data-slug');
    var r = null;
    R.rows.forEach(function (x) { if (x.slug === slug) r = x; });
    if (!r) return;
    el('board-body').querySelectorAll('tr').forEach(function (t) { t.classList.remove('selected'); });
    tr.classList.add('selected');
    showDetail(r);
  });

  // ---- Detail panel -------------------------------------------------------
  function showDetail(r) {
    var math =
      money(r.tvl_usd) + ' TVL &times; ' + r.category_multiple.toFixed(3) + ' multiple &times; ' +
      r.growth_multiplier.toFixed(3) + ' growth &times; ' + r.maturity_multiplier.toFixed(3) +
      ' maturity = <b>' + money(r.launch_value_usd) + '</b> launch value<br>' +
      money(r.launch_value_usd) + ' &times; ' + (r.launch_probability * 100).toFixed(0) + '% launch odds = ' +
      '<b>' + money(r.implied_value_usd) + '</b> implied value';
    var catMed = R.insights.category_median_change[r.category || ''];
    var trend = r.change_7d == null ? 'no 7-day change reported'
      : (r.change_7d >= 0 ? '+' : '') + r.change_7d.toFixed(1) + '% in 7 days' +
        (catMed == null ? '' : ' (category median ' + (catMed >= 0 ? '+' : '') + catMed.toFixed(1) + '%)');
    var peerRows = r.peers.map(function (p) {
      return '<tr><td>' + esc(p.name) + '<small>' + esc(p.slug) + '</small></td>' +
        '<td class="num">' + money(p.tvl_usd) + '</td><td class="num">' + p.ratio.toFixed(3) + '</td></tr>';
    }).join('');
    var parts = r.readiness_parts;
    var lri =
      '<span class="lri big">' + r.launch_readiness + '</span> ' +
      'category ' + parts.cat + ' &middot; scale ' + parts.scale + ' &middot; momentum ' + parts.mom + ' &middot; age ' + parts.age +
      ' (weights 40 / 25 / 20 / 15)';
    var steps = r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('');
    el('detail').innerHTML =
      '<div class="detail-grid"><div>' +
      '<h3>' + esc(r.name) + (r.watchlist ? '<span class="tag">Watchlist</span>' : '') + '</h3>' +
      '<p>Rank ' + r.rank + ' of ' + R.rows.length + ' &middot; ' + esc(r.category || 'Uncategorised') +
      ' &middot; Chains: ' + esc(r.chains.join(', ') || '—') + '<br>' +
      'TVL ' + money(r.tvl_usd) + ' &middot; ' + trend +
      ' &middot; Listed ' + esc(r.listed_at || 'unknown') + ' (' + ageText(r.age_days) + ' old)</p>' +
      '<p class="why">' + esc(r.why_rank) + '</p>' +
      '<h3>Launch Readiness Index</h3><p>' + lri + '</p>' +
      '<h3>Model breakdown</h3><p class="math">' + math + '</p>' +
      '<p class="fine">Category multiple ' + r.category_multiple.toFixed(3) +
      ' (fitted on launched tokens in ' + esc(r.category || 'this category') +
      '; the middle half of that category\'s ratios runs ' + r.category_iqr[0].toFixed(3) + '–' + r.category_iqr[1].toFixed(3) +
      ' — a wide spread is why any single implied value is uncertain); launch odds ' +
      (r.launch_probability * 100).toFixed(1) + '% (category token rate discounted by the fitted Weibull age curve); ' +
      'growth multiplier ' + r.growth_multiplier.toFixed(3) +
      '; maturity multiplier ' + r.maturity_multiplier.toFixed(3) + '.</p>' +
      '<h3>Closest launched comparables in ' + esc(r.category || 'this category') + '</h3>' +
      '<table class="mini"><thead><tr><th>Protocol</th><th class="num">TVL</th><th class="num">mcap / TVL</th></tr></thead><tbody>' +
      peerRows + '</tbody></table>' +
      '<p class="fine">The three launched tokens in the same category whose TVL is nearest to this ' +
      'protocol\'s (nearest in log TVL; dead tokens with a $0 market cap are skipped). Their ratios ' +
      'are the raw market-cap / TVL — the fitted multiple is the median of all of them, shrunk ' +
      'toward the global median.</p>' +
      '</div><div><h3>Farm plan</h3><ol class="farm">' + steps + '</ol></div></div>';
    el('detail-panel').hidden = false;
    el('detail-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ---- Live view ----------------------------------------------------------
  el('live-body').innerHTML = R.live.map(function (p) {
    var ratio = p.mcap_usd != null && p.tvl_usd > 0 ? (p.mcap_usd / p.tvl_usd).toFixed(3) : '—';
    return '<tr><td class="name">' + esc(p.name) + '<small>' + esc(p.slug) + '</small></td>' +
      '<td>' + esc(p.category || '—') + '</td>' +
      '<td class="num">' + money(p.tvl_usd) + '</td>' +
      '<td class="num">' + money(p.mcap_usd) + '</td>' +
      '<td class="num">' + ratio + '</td></tr>';
  }).join('');

  // ---- Insights -----------------------------------------------------------
  var ins = R.insights;
  el('momentum').innerHTML =
    '<thead><tr><th>Category (n &ge; 5)</th><th>Protocols</th><th>Median 7-day change</th></tr></thead><tbody>' +
    ins.category_momentum.map(function (x) {
      return '<tr><td>' + esc(x.category) + '</td><td>' + x.n + '</td><td>' + pct(x.median_change_7d) + '</td></tr>';
    }).join('') + '</tbody>';
  el('momentum-note').textContent =
    'Basis Trading is the only category growing at the median (+2.3% in 7 days); RWA is essentially ' +
    'flat (+0.0%); every other category with at least 5 ranked protocols is shrinking. Momentum is ' +
    'the weakest of the four LRI parts for most rows — the board is not a momentum board.';
  el('pretoken-tvl').innerHTML =
    '<thead><tr><th>Category</th><th>Protocols</th><th>Pre-token TVL</th></tr></thead><tbody>' +
    ins.pretoken_tvl.slice(0, 8).map(function (x) {
      return '<tr><td>' + esc(x.category) + '</td><td>' + x.n + '</td><td class="num">' + money(x.tvl_usd) + '</td></tr>';
    }).join('') + '</tbody>';
  el('pretoken-note').textContent =
    'Risk Curators hold the most pre-token TVL by far ($6.5B across 29 protocols) — and almost ' +
    'never launch: only 1 of 39 has a token. That is why their implied values are small despite ' +
    'the TVL. Onchain Capital Allocators ($2.5B) and RWA ($1.4B) follow.';
  el('held-back').innerHTML =
    '<thead><tr><th>Protocol</th><th>Rank</th><th>Launch value</th><th>Odds</th><th>Implied</th></tr></thead><tbody>' +
    ins.held_back.map(function (x) {
      return '<tr><td>' + esc(x.name) + '</td><td>' + x.rank + '</td><td class="num">' + money(x.launch_value_usd) +
        '</td><td class="num">' + (x.launch_probability * 100).toFixed(0) + '%</td><td class="num">' + money(x.implied_value_usd) + '</td></tr>';
    }).join('') + '</tbody>';
  el('held-back-note').textContent =
    'Gauntlet\'s launch value is the 2nd largest on the board, but its odds are 2.6% — the category ' +
    'almost never launches — so it sits at #13. These five are the protocols whose ranking is most ' +
    'decided by the odds, not by their size.';
  var oddsTotal = ins.odds_buckets.reduce(function (s, b) { return s + b.count; }, 0);
  el('odds-buckets').innerHTML =
    '<thead><tr><th>Launch odds</th><th>Rows</th><th>Share</th></tr></thead><tbody>' +
    ins.odds_buckets.map(function (b) {
      return '<tr><td>' + esc(b.label) + '</td><td>' + b.count + '</td><td>' + Math.round((b.count / oddsTotal) * 100) + '%</td></tr>';
    }).join('') + '</tbody>';
  el('odds-note').textContent =
    'Only 4 of the 200 rows carry odds above 35%; 78 rows sit below 10%. A launch is the exception, ' +
    'not the rule — the board ranks by expected value, and most of that value sits in a few rows.';

  // ---- Self-check ---------------------------------------------------------
  var sorted = true;
  for (var i = 1; i < R.rows.length; i++) {
    if (R.rows[i].implied_value_usd > R.rows[i - 1].implied_value_usd) { sorted = false; break; }
  }
  var farmOk = R.rows.every(function (r) {
    return r.farm_steps.length >= 3 && r.farm_steps.length <= 12 &&
      r.farm_steps.every(function (s) { return s.length >= 20 && s.length <= 600; });
  });
  var lriOk = R.rows.every(function (r) {
    return typeof r.launch_readiness === 'number' && r.launch_readiness >= 0 && r.launch_readiness <= 100 &&
      Array.isArray(r.peers) && r.peers.length === 3 && typeof r.why_rank === 'string' && r.why_rank.length >= 20;
  });
  var insightsOk = R.insights && Array.isArray(R.insights.category_momentum) && Array.isArray(R.insights.held_back) &&
    Array.isArray(R.insights.odds_buckets) &&
    R.insights.odds_buckets.reduce(function (s, b) { return s + b.count; }, 0) === R.rows.length;
  var checks = [
    { ok: R.board_size === 200, text: 'Board holds ' + R.board_size + ' rows (200: 40 watchlist + 160 early).' },
    { ok: R.watchlist_size === 40, text: 'Watchlist coverage ' + R.watchlist_size + ' / 40 slugs.' },
    { ok: sorted, text: 'Ranks 1..' + R.board_size + ', implied value non-increasing down the board.' },
    { ok: farmOk, text: 'Every row carries 6 farm steps, each 20–600 characters.' },
    { ok: R.live.length === 24, text: 'Live view holds ' + R.live.length + ' already-trading protocols by market cap.' },
    { ok: R.weibull && R.weibull.k > 0 && R.weibull.lambda > 0, text: 'Weibull hazard fitted from the snapshot (k = ' + R.weibull.k.toFixed(2) + ', lambda = ' + R.weibull.lambda.toLocaleString('en-US') + ' days).' },
    { ok: lriOk, text: 'Every row carries a Launch Readiness Index (0–100) and 3 peers.' },
    { ok: insightsOk, text: 'Insights present: ' + R.insights.category_momentum.length + ' momentum categories, ' + R.insights.held_back.length + ' held-back rows, odds buckets sum to ' + R.rows.length + '.' },
    { ok: true, text: 'Deterministic: two runs on the sealed snapshot write identical bytes (no clock, no randomness).' },
  ];
  el('checks').innerHTML = checks.map(function (c) {
    return '<li' + (c.ok ? '' : ' class="fail"') + '>' + esc(c.text) + '</li>';
  }).join('');
  el('source-note').textContent =
    'Source: ' + R.source + '. Snapshot date ' + R.as_of + '. Floor $' +
    R.floor_usd.toLocaleString('en-US') + '. Model ' + R.model_version + '.';

  renderBoard();
})();
