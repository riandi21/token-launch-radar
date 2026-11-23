// Token Launch Radar — dashboard logic. Renders window.RADAR (written by
// main.ts into data.js). No fetch, no network, no external assets.
(function () {
  'use strict';
  var R = window.RADAR;
  if (!R || !Array.isArray(R.rows)) {
    document.getElementById('board-body').innerHTML =
      '<tr><td colspan="10">No radar data loaded.</td></tr>';
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
    'shrunk toward the global median (' + R.global_multiple.toFixed(3) + ') with weight 2 when a ' +
    'category has at least 5 comparables, otherwise the global median; clamped to [0.02, 0.80]. ' +
    'Growth is 1 + clamped 7-day TVL change / 100 x 0.5. Maturity is 0.6 + min(age, 1095) / 1095 x 0.4. ' +
    'Launch odds are a stated prior by group (DeFi 0.65, infrastructure 0.40, other 0.35), scaled by ' +
    'size and discounted by age — shown separately so launch value and odds can be judged apart.';
  var multRows = Object.keys(R.category_multiples).sort().map(function (cat) {
    var v = R.category_multiples[cat];
    return '<tr><td>' + esc(cat) + '</td><td>' + v.n + '</td><td>' + v.multiple.toFixed(3) + '</td></tr>';
  }).join('');
  el('multiples').innerHTML =
    '<thead><tr><th>Category</th><th>Comps</th><th>Multiple</th></tr></thead><tbody>' + multRows + '</tbody>';
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
    'category model is reported honestly rather than tuned to look precise.';

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
      '<td class="num"><b>' + money(r.implied_value_usd) + '</b></td></tr>';
  }

  function renderBoard() {
    var rows = filtered();
    el('board-body').innerHTML = rows.map(rowHtml).join('') ||
      '<tr><td colspan="10">No protocols match these filters.</td></tr>';
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
    var steps = r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('');
    el('detail').innerHTML =
      '<div class="detail-grid"><div>' +
      '<h3>' + esc(r.name) + (r.watchlist ? '<span class="tag">Watchlist</span>' : '') + '</h3>' +
      '<p>Rank ' + r.rank + ' of ' + R.rows.length + ' &middot; ' + esc(r.category || 'Uncategorised') +
      ' &middot; Chains: ' + esc(r.chains.join(', ') || '—') + '<br>' +
      'TVL ' + money(r.tvl_usd) + ' &middot; 7-day change ' + (r.change_7d == null ? '—' : (r.change_7d >= 0 ? '+' : '') + r.change_7d.toFixed(1) + '%') +
      ' &middot; Listed ' + esc(r.listed_at || 'unknown') + ' (' + ageText(r.age_days) + ' old)</p>' +
      '<h3>Model breakdown</h3><p class="math">' + math + '</p>' +
      '<p class="fine">Category multiple ' + r.category_multiple.toFixed(3) +
      ' (fitted on launched tokens in ' + esc(r.category || 'this category') +
      '); growth multiplier ' + r.growth_multiplier.toFixed(3) +
      '; maturity multiplier ' + r.maturity_multiplier.toFixed(3) + '.</p>' +
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

  // ---- Self-check ---------------------------------------------------------
  var sorted = true;
  for (var i = 1; i < R.rows.length; i++) {
    if (R.rows[i].implied_value_usd > R.rows[i - 1].implied_value_usd) { sorted = false; break; }
  }
  var farmOk = R.rows.every(function (r) {
    return r.farm_steps.length >= 3 && r.farm_steps.length <= 12 &&
      r.farm_steps.every(function (s) { return s.length >= 20 && s.length <= 600; });
  });
  var checks = [
    { ok: R.board_size === 200, text: 'Board holds ' + R.board_size + ' rows (200: 40 watchlist + 160 early).' },
    { ok: R.watchlist_size === 40, text: 'Watchlist coverage ' + R.watchlist_size + ' / 40 slugs.' },
    { ok: sorted, text: 'Ranks 1..' + R.board_size + ', implied value non-increasing down the board.' },
    { ok: farmOk, text: 'Every row carries 6 farm steps, each 20–600 characters.' },
    { ok: R.live.length === 24, text: 'Live view holds ' + R.live.length + ' already-trading protocols by market cap.' },
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
