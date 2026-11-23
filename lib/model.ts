// Token Launch Radar — ranking model (radar-4).
//
// One sealed DefiLlama snapshot in, one ranked board out. Every number the
// model uses is fitted from that snapshot at run time: nothing about any
// single protocol is hard-coded, and no protocol name is ever special-cased.
//
//   implied value = TVL x category multiple x growth x maturity x launch odds
//
// The category multiple is the median market-cap/TVL ratio of launched
// tokens in the same category, shrunk toward the global median and clamped.
// Launch odds are the category's own empirical token rate — the share of
// protocols in that category that already have a token, shrunk toward the
// global token rate — discounted by age. The snapshot IS the launch
// history: a cross-section of 758 protocols, 144 of which carry a token.
// Farm plans are derived from each protocol's own category, chains,
// TVL and age.
//
// What v3 changed from v2 (radar-3), and why:
// - The age discount is now FITTED, not hand-set. v2 used fixed steps
//   (1.00 <180d, 0.95, 0.90, 0.80, 0.70 >3y). v3 fits a Weibull hazard
//   model to the snapshot itself: P(token by age a) = 1 - exp(-(a/lambda)^k).
//   The fit gives k = 0.76, lambda = 6500 days — k < 1 means a DECREASING
//   hazard, i.e. a protocol that survives years without launching is less
//   likely to launch next year. The discount is the 1-year forward launch
//   probability relative to a 90-day-old protocol, so it is smooth (0.93 at
//   180d, 0.83 at 1y, 0.74 at 2y, 0.68 at 3y, 0.57 at ~7y) instead of steps.
//   It is fitted on the train split for validation and on all protocols for
//   the board, exactly like the launch rates. On the has-token classifier it
//   scores AUC 0.519, Brier 0.1765 — slightly better than v2's hand-set
//   steps (0.501 / 0.1751) on AUC, slightly worse on Brier, and worse than
//   no discount at all (0.524 / 0.1704), which is expected: the classifier
//   asks who HAS a token now, and in the cross-section older protocols are
//   MORE likely to have one (token rate 0.034 under 180d, 0.294 over 3y).
//   The discount answers the forward question — who WILL launch next —
//   and the Weibull fit is the data's own answer to it, not a hand-set
//   guess. The top 10 barely moves (Veda, USDD, OnRe stay 1-2-3; the
//   biggest mover anywhere is 14 ranks), so the discount is a refinement,
//   not a reshuffle.
// - The mcap estimator is UNCHANGED (median multiple, shrink weight 5).
//   v3 tested trimmed means (10%: 118.2%, 20%: 90.3% — both worse),
//   a median (L1) regression on log mcap (solver returns non-finite
//   coefficients, the same numerical failure as v2; v1 already showed the
//   log-log regression does not beat flat), a primary-chain multiple (no
//   better than category, and the geometric mean of the two loses on
//   walk-forward), a growth interaction (slopes ~0, no improvement), and
//   an age effect on the multiple (no monotone trend across age bands).
//   The median multiple is at its ceiling on this snapshot: 84.1% median
//   error, 35.5% within 2x on the hash split, still short of flat 0.25
//   (82.7%). v3 reports that honestly instead of tuning the multiple to
//   look precise. change_1m was never tested — the snapshot carries no
//   change_1d or change_1m field, only change_7d (754/758 non-null).
// - NEW: the Launch Readiness Index (LRI), a 0-100 score answering a
//   different question from implied value: how ready is this protocol to
//   launch — category launch rate (40%), TVL scale (25%), 7-day momentum
//   (20%), age/maturity (15%). It is a separate column and sort on the
//   board. It correlates 0.78 with implied-value rank (Spearman; same
//   ingredients, different question) and surfaces protocols the value
//   ranking buries (Shift DeFi is implied #30 but LRI #3; RIF ON CHAIN
//   #54, LRI #2; Manta CeDeFi #27, LRI #4). Reported as a lens, not a
//   prediction.
// - NEW: category momentum (median 7-day change of pre-token protocols
//   per category), pre-token TVL by category, the "held back by odds"
//   list (biggest launch value outside the implied top 10), and an
//   odds-distribution summary — all computed from the snapshot at runtime.
// - NEW in the detail view: the protocol's 7-day change against its
//   category's median, its three closest comparables in the same category
//   (by TVL), and a plain-language "why this rank" line naming the factor
//   that drives the value.
//
// Node 22, standard library only. Fully deterministic: same snapshot in,
// same board out, byte for byte.

import { createHash } from 'node:crypto';

/** One protocol entry, exactly the ten fields the sealed snapshot carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole sealed snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The constants that shape the model (the fitted multiples, rates and hazard are computed, not constants). */
export interface ModelConstants {
  shrink_weight: number;
  multiple_floor: number;
  multiple_cap: number;
  growth_weight: number;
  growth_floor: number;
  growth_cap: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
  prob_shrink_weight: number;
  prob_floor: number;
  prob_cap: number;
  /** Weibull hazard grid for the age discount: k from k_min to k_max in k_step; lambda from lam_min to lam_max in lam_step. */
  weibull_k_min: number;
  weibull_k_max: number;
  weibull_k_step: number;
  weibull_lam_min: number;
  weibull_lam_max: number;
  weibull_lam_step: number;
  /** Forward horizon, in days, for the age discount (1-year forward launch probability). */
  weibull_horizon_days: number;
  /** Reference age, in days, the age factor is normalised against (a 90-day-old protocol scores 1.0). */
  weibull_ref_days: number;
  /** Launch Readiness Index weights (category rate, TVL scale, 7-day momentum, age), summing to 100. */
  lri_cat_weight: number;
  lri_scale_weight: number;
  lri_mom_weight: number;
  lri_age_weight: number;
}

export const MODEL: ModelConstants = {
  shrink_weight: 5,
  multiple_floor: 0.02,
  multiple_cap: 0.8,
  growth_weight: 0.5,
  growth_floor: -60,
  growth_cap: 120,
  maturity_floor: 0.6,
  maturity_range: 0.4,
  maturity_days: 1095,
  unknown_age_days: 365,
  prob_shrink_weight: 2,
  prob_floor: 0.02,
  prob_cap: 0.9,
  weibull_k_min: 0.4,
  weibull_k_max: 1.6,
  weibull_k_step: 0.04,
  weibull_lam_min: 500,
  weibull_lam_max: 15000,
  weibull_lam_step: 500,
  weibull_horizon_days: 365,
  weibull_ref_days: 90,
  lri_cat_weight: 40,
  lri_scale_weight: 25,
  lri_mom_weight: 20,
  lri_age_weight: 15,
};

export const MODEL_VERSION = 'radar-4';
export const EARLY_ROWS = 160;
export const LIVE_ROWS = 24;

/** One board row: the ranked pre-token protocol and everything behind its value. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  launch_value_usd: number;
  launch_probability: number;
  /** Launch Readiness Index, 0-100: how ready this protocol is to launch (a different question from implied value). */
  launch_readiness: number;
  readiness_parts: { cat: number; scale: number; mom: number; age: number };
  /** The three closest launched comparables in the same category, by TVL proximity. */
  peers: Array<{ slug: string; name: string; tvl_usd: number; ratio: number }>;
  /** Plain-language line naming the factor that drives this rank. */
  why_rank: string;
  model: {
    category_multiple: number;
    category_iqr: [number, number];
    growth_multiplier: number;
    maturity_multiplier: number;
    growth_pct: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** A board row trimmed for the dashboard's data file. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  launch_value_usd: number;
  launch_probability: number;
  launch_readiness: number;
  readiness_parts: { cat: number; scale: number; mom: number; age: number };
  peers: Array<{ slug: string; name: string; tvl_usd: number; ratio: number }>;
  why_rank: string;
  category_multiple: number;
  category_iqr: [number, number];
  growth_multiplier: number;
  maturity_multiplier: number;
  watchlist: boolean;
  farm_steps: string[];
}

/** A protocol that already trades: the market's own value, never a model guess. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** One held-out model's score: median absolute error and share inside a 2x band. */
export interface ValidationScore {
  name: string;
  median_abs_pct_error: number;
  within_2x_pct: number;
  test_n: number;
}

/** One launch-prior model's score as a classifier: does it predict which protocols carry a token? */
export interface PriorScore {
  name: string;
  brier: number;
  log_loss: number;
  auc: number;
  test_n: number;
  positives: number;
}

/** One walk-forward split: train on protocols listed before the cutoff, test after it. */
export interface WalkForwardSplit {
  cutoff: string;
  train_n: number;
  test_n: number;
  models: ValidationScore[];
}

/** The held-out split: train/test sizes plus one score per model compared. */
export interface HeldOutValidation {
  split: string;
  train_n: number;
  test_n: number;
  models: ValidationScore[];
  walk_forward: WalkForwardSplit[];
  prior: PriorScore[];
}

/** Everything the dashboard renders, from one snapshot. */
export interface RadarData {
  as_of: string;
  generated_at: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  global_multiple: number;
  category_multiples: Record<string, { n: number; multiple: number; iqr: [number, number] }>;
  global_launch_rate: number;
  launch_rates: Record<string, { n: number; rate: number }>;
  /** The fitted Weibull hazard (k, lambda) behind the age discount, plus its log-likelihood. */
  weibull: { k: number; lambda: number; loglik: number };
  validation: HeldOutValidation;
  /** Cross-cutting insights, all computed from the snapshot at runtime. */
  insights: {
    /** Median 7-day TVL change of pre-token protocols per category (n >= 5), best first. */
    category_momentum: Array<{ category: string; n: number; median_change_7d: number }>;
    /** Pre-token TVL per category, largest first. */
    pretoken_tvl: Array<{ category: string; n: number; tvl_usd: number }>;
    /** Biggest launch value outside the implied top 10 — the odds are holding these back. */
    held_back: Array<{ slug: string; name: string; rank: number; launch_value_usd: number; launch_probability: number; implied_value_usd: number }>;
    /** Launch-odds distribution across the 200 rows: counts in five buckets. */
    odds_buckets: Array<{ label: string; count: number }>;
    /** Median 7-day change of pre-token protocols in each board row's category, for the detail view. */
    category_median_change: Record<string, number>;
  };
  rows: RadarRow[];
  live: LiveRow[];
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n));
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** The median of a non-empty list of numbers. */
export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** The q-th quantile (0 <= q <= 1) of a non-empty list, by linear interpolation. */
export function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0]!;
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

/** Whole days from `listedAt` to `asOf`, at least zero; null when the date is missing or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** A protocol records no token at all: no market cap, no CoinGecko id, no symbol. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol is one the early radar may rank: pre-token, not an exchange, not versioned. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

// ---------------------------------------------------------------- fitted multiples

/** A launched token with a TVL to compare against: the comparables the multiples are fitted on. */
export function comparables(protocols: SnapshotProtocol[]): SnapshotProtocol[] {
  return protocols.filter((p) => p.mcap_usd !== null && p.tvl_usd > 0);
}

/**
 * The fitted multiples: the global median mcap/TVL over all comparables, and
 * one multiple per category — the category median shrunk toward the global
 * with weight `shrink_weight` when the category has at least 5 comparables,
 * the global median otherwise — all clamped to [multiple_floor, multiple_cap].
 * Each category also carries the interquartile range of its ratios, so the
 * dashboard can show how wide the multiple's uncertainty is.
 */
export function fitMultiples(
  protocols: SnapshotProtocol[],
  model: ModelConstants = MODEL,
): { global: number; globalIqr: [number, number]; byCategory: Map<string, { n: number; multiple: number; iqr: [number, number] }> } {
  const comps = comparables(protocols);
  const ratios = comps.map((p) => p.mcap_usd! / p.tvl_usd);
  const global = median(ratios);
  const globalIqr: [number, number] = [round4(quantile(ratios, 0.25)), round4(quantile(ratios, 0.75))];
  const byCat = new Map<string, number[]>();
  for (const p of comps) {
    const key = p.category ?? 'Unknown';
    const list = byCat.get(key);
    if (list === undefined) byCat.set(key, [p.mcap_usd! / p.tvl_usd]);
    else list.push(p.mcap_usd! / p.tvl_usd);
  }
  const byCategory = new Map<string, { n: number; multiple: number; iqr: [number, number] }>();
  for (const [cat, rs] of [...byCat.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const n = rs.length;
    const raw = n >= 5 ? (n * median(rs) + model.shrink_weight * global) / (n + model.shrink_weight) : global;
    const iqr: [number, number] = n >= 5
      ? [round4(quantile(rs, 0.25)), round4(quantile(rs, 0.75))]
      : globalIqr;
    byCategory.set(cat, { n, multiple: round4(clamp(raw, model.multiple_floor, model.multiple_cap)), iqr });
  }
  return { global: round4(global), globalIqr, byCategory };
}

/** A category's fitted multiple, or the global median when the category has too few comparables. */
export function categoryMultiple(
  category: string | null,
  fitted: { global: number; byCategory: Map<string, { n: number; multiple: number; iqr: [number, number] }> },
): number {
  if (category === null) return fitted.global;
  return fitted.byCategory.get(category)?.multiple ?? fitted.global;
}

/** A category's multiple interquartile range, or the global one when the category has too few comparables. */
export function categoryIqr(
  category: string | null,
  fitted: { globalIqr: [number, number]; byCategory: Map<string, { n: number; multiple: number; iqr: [number, number] }> },
): [number, number] {
  if (category === null) return fitted.globalIqr;
  return fitted.byCategory.get(category)?.iqr ?? fitted.globalIqr;
}

// ---------------------------------------------------------------- growth, maturity, odds

/** The growth multiplier from the 7-day TVL change, clamped to [-60, +120]%. Null change counts as flat. */
export function growthMultiplier(change7d: number | null, model: ModelConstants = MODEL): { pct: number; mult: number } {
  const raw = typeof change7d === 'number' && Number.isFinite(change7d) ? change7d : 0;
  const pct = clamp(raw, model.growth_floor, model.growth_cap);
  return { pct, mult: 1 + (pct / 100) * model.growth_weight };
}

/** The maturity multiplier from age in days: 0.6 for a newborn, 1.0 at three years. Unknown age is one year. */
export function maturityMultiplier(age: number | null, model: ModelConstants = MODEL): number {
  const days = age ?? model.unknown_age_days;
  return model.maturity_floor + (Math.min(Math.max(days, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;
}

// ---------------------------------------------------------------- fitted Weibull age discount

/** A fitted Weibull hazard: P(token by age a) = 1 - exp(-(a/lambda)^k), plus the fit's log-likelihood. */
export interface WeibullFit {
  k: number;
  lambda: number;
  loglik: number;
}

/**
 * The Weibull hazard fitted to the snapshot itself, by current-status
 * maximum likelihood: a protocol that carries a token contributes
 * log P(token by its age), one that does not contributes log P(no token
 * by its age). Grid-searched over the k/lambda ranges in MODEL, so the
 * fit is deterministic and reproducible. Ages under one day count as one
 * day (a protocol listed today has had no time to launch, but log(0)
 * would sink the fit). Protocols with no listing date are left out.
 * Fewer than 10 usable protocols falls back to a neutral shape (k = 1,
 * lambda = 10000) rather than a fit on noise.
 */
export function fitWeibull(
  protocols: SnapshotProtocol[],
  asOf: string,
  model: ModelConstants = MODEL,
): WeibullFit {
  const rows: Array<{ age: number; token: boolean }> = [];
  for (const p of protocols) {
    const age = ageDays(asOf, p.listed_at);
    if (age === null) continue;
    rows.push({ age: Math.max(age, 1), token: p.mcap_usd !== null });
  }
  if (rows.length < 10) return { k: 1, lambda: 10000, loglik: 0 };
  let best: WeibullFit = { k: 1, lambda: 10000, loglik: -Infinity };
  for (let k = model.weibull_k_min; k <= model.weibull_k_max + 1e-9; k += model.weibull_k_step) {
    for (let lam = model.weibull_lam_min; lam <= model.weibull_lam_max; lam += model.weibull_lam_step) {
      let ll = 0;
      for (const r of rows) {
        const x = (r.age / lam) ** k;
        if (r.token) {
          const t = 1 - Math.exp(-x);
          ll += t > 0 ? Math.log(t) : Math.log(x);
        } else {
          ll -= x;
        }
      }
      if (ll > best.loglik) best = { k: round4(k), lambda: lam, loglik: Math.round(ll * 100) / 100 };
    }
  }
  return best;
}

/** The Weibull survival at an age: P(no token by age a). */
export function weibullSurvival(age: number, fit: WeibullFit): number {
  return Math.exp(-(((Math.max(age, 0)) / fit.lambda) ** fit.k));
}

/** The 1-year forward launch probability at an age: P(launch in the next year | survived to a). */
export function weibullForward(age: number, fit: WeibullFit, horizonDays: number): number {
  const s0 = weibullSurvival(age, fit);
  if (s0 <= 0) return 0;
  return 1 - weibullSurvival(age + horizonDays, fit) / s0;
}

/**
 * The age discount from the fitted hazard: the 1-year forward launch
 * probability at this age, relative to a 90-day-old protocol (which scores
 * 1.0). Young protocols are not boosted above 1.0 — the discount only
 * ever lowers odds. Unknown age scores 0.85, the same neutral value v2
 * used. k < 1 means a decreasing hazard: surviving years without launching
 * lowers the next year's odds.
 */
export function ageDiscount(age: number | null, fit: WeibullFit, model: ModelConstants = MODEL): number {
  if (age === null) return 0.85;
  const ref = weibullForward(model.weibull_ref_days, fit, model.weibull_horizon_days);
  if (ref <= 0) return 1;
  return clamp(weibullForward(age, fit, model.weibull_horizon_days) / ref, 0, 1);
}

/** The v2 hand-set age steps, kept only so the prior test can score them against the fitted discount. */
export function ageFactorSteps(age: number | null): number {
  return age === null ? 0.85 : age < 180 ? 1.0 : age < 365 ? 0.95 : age < 730 ? 0.9 : age < 1095 ? 0.8 : 0.7;
}

/**
 * The fitted launch rates: the global share of protocols that carry a token,
 * and one rate per category — the category's token share shrunk toward the
 * global with weight `prob_shrink_weight`. The snapshot is the launch
 * history: a cross-section of 758 protocols, 144 of which carry a token.
 * Categories with few protocols shrink most, so a 1-of-1 category does not
 * claim a 100% launch rate.
 */
export function fitLaunchRates(
  protocols: SnapshotProtocol[],
  model: ModelConstants = MODEL,
): { global: number; byCategory: Map<string, { n: number; rate: number }> } {
  const byCat = new Map<string, { t: number; k: number }>();
  let tot = 0;
  let tok = 0;
  for (const p of protocols) {
    const key = p.category ?? 'Unknown';
    const e = byCat.get(key) ?? { t: 0, k: 0 };
    e.t++;
    tot++;
    if (p.mcap_usd !== null) {
      e.k++;
      tok++;
    }
    byCat.set(key, e);
  }
  const global = tot > 0 ? tok / tot : 0.19;
  const byCategory = new Map<string, { n: number; rate: number }>();
  for (const [cat, e] of [...byCat.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const rate = (e.k + model.prob_shrink_weight * global) / (e.t + model.prob_shrink_weight);
    byCategory.set(cat, { n: e.t, rate: round4(rate) });
  }
  return { global: round4(global), byCategory };
}

/** A category's fitted launch rate, or the global token rate when the category is unknown. */
export function launchRate(
  category: string | null,
  fitted: { global: number; byCategory: Map<string, { n: number; rate: number }> },
): number {
  if (category === null) return fitted.global;
  return fitted.byCategory.get(category)?.rate ?? fitted.global;
}

/**
 * The launch odds: the category's empirical token rate, discounted by age.
 * The age discount is now the fitted Weibull hazard (see ageDiscount), not
 * a hand-set table: the snapshot's own launch history says how the odds of
 * launching next year fall with age. No size factor: token rate by TVL
 * quartile is flat (smallest quartile 0.249, largest 0.168), so a bigger
 * TVL does not mean higher odds. TVL and age already enter through launch
 * value, so the odds carry only the category rate and the age discount —
 * no double-counting.
 */
export function launchProbability(
  p: SnapshotProtocol,
  age: number | null,
  fitted: { global: number; byCategory: Map<string, { n: number; rate: number }> },
  model: ModelConstants = MODEL,
  weibull: WeibullFit,
): number {
  const base = launchRate(p.category, fitted);
  return round4(clamp(base * ageDiscount(age, weibull, model), model.prob_floor, model.prob_cap));
}

/** The launch odds with v2's hand-set age steps, kept only for the prior test's comparison. */
export function launchProbabilitySteps(
  p: SnapshotProtocol,
  age: number | null,
  fitted: { global: number; byCategory: Map<string, { n: number; rate: number }> },
  model: ModelConstants = MODEL,
): number {
  const base = launchRate(p.category, fitted);
  return round4(clamp(base * ageFactorSteps(age), model.prob_floor, model.prob_cap));
}

/** One protocol's full valuation: launch value before odds, implied value after. */
export function scoreProtocol(
  p: SnapshotProtocol,
  asOf: string,
  fitted: { global: number; globalIqr: [number, number]; byCategory: Map<string, { n: number; multiple: number; iqr: [number, number] }> },
  rates: { global: number; byCategory: Map<string, { n: number; rate: number }> },
  model: ModelConstants = MODEL,
  weibull: WeibullFit,
): {
  age: number | null;
  categoryMultiple: number;
  categoryIqr: [number, number];
  growthPct: number;
  growthMult: number;
  maturityMult: number;
  launchValue: number;
  probability: number;
  implied: number;
} {
  const age = ageDays(asOf, p.listed_at);
  const catMult = categoryMultiple(p.category, fitted);
  const catIqr = categoryIqr(p.category, fitted);
  const { pct: growthPct, mult: growthMult } = growthMultiplier(p.change_7d, model);
  const maturityMult = maturityMultiplier(age, model);
  const launchValue = p.tvl_usd * catMult * growthMult * maturityMult;
  const probability = launchProbability(p, age, rates, model, weibull);
  const implied = Math.max(1, Math.round(launchValue * probability));
  return { age, categoryMultiple: catMult, categoryIqr: catIqr, growthPct, growthMult, maturityMult, launchValue, probability, implied };
}

// ---------------------------------------------------------------- readiness, peers, why

/** A dollar amount in short form ($1.2B, $45M, $3K) for plain-language lines. */
function fmtUsd(n: number): string {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${Math.round(n)}`;
}

/**
 * The Launch Readiness Index (LRI), 0-100: how ready this protocol is to
 * launch — a different question from implied value, which asks what a
 * launch would be worth. Four parts, each scored 0-100 and combined by
 * the weights in MODEL: category launch rate (40%), TVL scale (25%),
 * 7-day momentum (20%), age/maturity (15%). Category and scale are
 * normalised against the board itself (the highest-rate category and the
 * log-TVL range of the ranked rows), so the index is a relative lens
 * within this board, not an absolute prediction.
 */
export function launchReadiness(
  p: SnapshotProtocol,
  s: { age: number | null; growthPct: number },
  rate: number,
  ctx: { maxRate: number; minLogTvl: number; maxLogTvl: number },
  model: ModelConstants = MODEL,
): { readiness: number; parts: { cat: number; scale: number; mom: number; age: number } } {
  const cat = ctx.maxRate > 0 ? (rate / ctx.maxRate) * 100 : 0;
  const logTvl = Math.log10(Math.max(p.tvl_usd, 1));
  const scale = ctx.maxLogTvl > ctx.minLogTvl ? ((logTvl - ctx.minLogTvl) / (ctx.maxLogTvl - ctx.minLogTvl)) * 100 : 50;
  const mom = ((clamp(s.growthPct, model.growth_floor, model.growth_cap) + 60) / 180) * 100;
  const ageDays = s.age ?? model.unknown_age_days;
  const age = (Math.min(Math.max(ageDays, 0), model.maturity_days) / model.maturity_days) * 100;
  const parts = { cat: Math.round(cat), scale: Math.round(scale), mom: Math.round(mom), age: Math.round(age) };
  const readiness = Math.round(
    (model.lri_cat_weight * cat + model.lri_scale_weight * scale + model.lri_mom_weight * mom + model.lri_age_weight * age) / 100,
  );
  return { readiness: clamp(readiness, 0, 100), parts };
}

/**
 * The three closest launched comparables in the same category, by TVL
 * proximity (nearest in log TVL first). If the category holds fewer than
 * three, the rest come from the closest comparables anywhere, so every
 * row still shows three peers. Dead tokens (market cap 0 — Multichain,
 * Phemex and 13 others in this snapshot) are skipped as peers: a $0
 * market cap says nothing about what a launch is worth. They still count
 * in the fitted multiples, exactly as in v2, so the validation numbers
 * stay comparable.
 */
export function peersFor(
  p: SnapshotProtocol,
  comps: SnapshotProtocol[],
): Array<{ slug: string; name: string; tvl_usd: number; ratio: number }> {
  const live = comps.filter((q) => (q.mcap_usd ?? 0) > 0);
  const dist = (q: SnapshotProtocol): number => Math.abs(Math.log10(q.tvl_usd / Math.max(p.tvl_usd, 1)));
  const same = live
    .filter((q) => q.category === p.category && q.slug !== p.slug)
    .sort((a, b) => dist(a) - dist(b));
  const picked = same.slice(0, 3);
  if (picked.length < 3) {
    const rest = live
      .filter((q) => q.slug !== p.slug && !picked.includes(q))
      .sort((a, b) => dist(a) - dist(b));
    picked.push(...rest.slice(0, 3 - picked.length));
  }
  return picked.map((q) => ({ slug: q.slug, name: q.name, tvl_usd: q.tvl_usd, ratio: round4(q.mcap_usd! / q.tvl_usd) }));
}

/**
 * The plain-language line naming the factor that drives a rank: the
 * category multiple, the 7-day TVL move, the launch odds, or the TVL
 * scale — whichever deviates furthest from the board's own centre.
 */
export function whyRank(
  row: { rank: number; category: string | null; tvl_usd: number; launch_probability: number },
  s: { categoryMultiple: number; growthPct: number },
  ctx: { globalMultiple: number; globalRate: number; medianTvl: number },
): string {
  const cands: Array<{ dev: number; text: string }> = [
    {
      dev: Math.abs(Math.log(s.categoryMultiple / ctx.globalMultiple)),
      text: s.categoryMultiple > ctx.globalMultiple
        ? `its ${row.category ?? 'category'} multiple of ${s.categoryMultiple.toFixed(3)} — ${(s.categoryMultiple / ctx.globalMultiple).toFixed(1)}x the ${ctx.globalMultiple.toFixed(3)} global median`
        : `its ${row.category ?? 'category'} multiple of ${s.categoryMultiple.toFixed(3)} — below the ${ctx.globalMultiple.toFixed(3)} global median`,
    },
    {
      dev: Math.abs(s.growthPct) / 50,
      text: s.growthPct >= 0
        ? `its +${s.growthPct.toFixed(1)}% 7-day TVL growth`
        : `its ${s.growthPct.toFixed(1)}% 7-day TVL fall`,
    },
    {
      dev: Math.abs(Math.log(row.launch_probability / ctx.globalRate)),
      text: row.launch_probability > ctx.globalRate
        ? `launch odds of ${(row.launch_probability * 100).toFixed(0)}% — above the ${(ctx.globalRate * 100).toFixed(1)}% average`
        : `launch odds of ${(row.launch_probability * 100).toFixed(0)}% — below the ${(ctx.globalRate * 100).toFixed(1)}% average`,
    },
    {
      dev: Math.abs(Math.log10(row.tvl_usd / ctx.medianTvl)) / 1.5,
      text: row.tvl_usd > ctx.medianTvl
        ? `a ${fmtUsd(row.tvl_usd)} TVL — well above the ${fmtUsd(ctx.medianTvl)} board median`
        : `a ${fmtUsd(row.tvl_usd)} TVL — well below the ${fmtUsd(ctx.medianTvl)} board median`,
    },
  ];
  cands.sort((a, b) => b.dev - a.dev);
  return `Ranked #${row.rank} mostly on ${cands[0]!.text}.`;
}

// ---------------------------------------------------------------- farm plans

/** The first chain a protocol names, or a neutral fallback. */
function firstChain(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** The capital one farm position needs, sized by the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols need a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** A rough gas band for one transaction on a chain, in dollars. */
function gasFor(chain: string): string {
  if (chain === 'Ethereum') return '$5-15';
  if (chain === 'Arbitrum' || chain === 'Optimism' || chain === 'Base') return '$0.10-1';
  if (chain === 'Solana') return 'about $0.01';
  return '$0.50-2';
}

/** The category's one action, in words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return `Lend $${capital} of USDC into ${name} on ${chain} and leave the supply position open.`;
  }
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) {
    return `Add $${capital} of liquidity to ${name}'s main pool on ${chain} and leave the position open.`;
  }
  if (c.includes('bridge')) {
    return `Bridge $${capital} of USDC through ${name} to ${chain} and use the bridged funds inside its app.`;
  }
  if (c.includes('stak') || c.includes('restak')) {
    return `Stake $${capital} of the chain asset through ${name} on ${chain} and keep the receipt token.`;
  }
  if (c.includes('perpetual') || c.includes('derivative')) {
    return `Deposit $${capital} of USDC as margin on ${name} and open one small hedged trade.`;
  }
  if (c.includes('prediction')) {
    return `Fund $${capital} on ${name} and take one position in an open market.`;
  }
  if (c.includes('curator') || c.includes('allocator')) {
    return `Deposit $${capital} into a ${name} vault on ${chain} and hold the vault shares.`;
  }
  if (c.includes('stable')) {
    return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  }
  if (c.includes('rwa')) {
    return `Mint $${capital} of ${name}'s yield-bearing asset on ${chain} and hold it.`;
  }
  if (c.includes('insurance')) {
    return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active.`;
  }
  return `Deposit $${capital} of USDC into ${name} on ${chain} and leave the position open.`;
}

/**
 * A protocol's farm plan: six steps, each one string carrying its action,
 * its cost and its time. Every step is derived from the protocol's own
 * record — category, chains, TVL, age — never from a hand-written list.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): string[] {
  const chain = firstChain(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasFor(chain);
  const steps = [
    `Set up a fresh wallet that holds only the funds you plan to farm, and store its seed phrase offline, never in a screenshot or a chat. Cost: under $3 of gas for the first transaction. Time: 5 minutes.`,
    `${actionFor(p, capital, chain)} Cost: $${capital} of capital plus about ${gas} of gas. Time: 15 minutes to set up.`,
    `Do one qualifying action each week — a deposit, a borrow, a swap or a vote — so the wallet shows steady activity instead of a single visit. Cost: about $2 of gas a week. Time: 10 minutes a week.`,
    `Hold the position for at least ${hold} days so the activity spans more than one snapshot window. Cost: the $${capital} stays in the position. Time: ${hold} days.`,
    `Before withdrawing, open the protocol's own points or rewards page and claim only from the link that page itself publishes — never from a link in a chat or a search ad. Cost: $0 to check. Time: 10 minutes.`,
    `Write the wallet address, the dates and every transaction hash in a farm log, so you can repeat the position or unwind it cleanly. Cost: $0. Time: 5 minutes.`,
  ];
  return steps.map((s) => (s.length > 600 ? `${s.slice(0, 596)}...` : s));
}

// ---------------------------------------------------------------- the board

/** The ranked board: every watchlist slug plus the strongest early rows, ranks 1..n. */
export function buildBoard(
  snapshot: Snapshot,
  fitted: { global: number; globalIqr: [number, number]; byCategory: Map<string, { n: number; multiple: number; iqr: [number, number] }> },
  rates: { global: number; byCategory: Map<string, { n: number; rate: number }> },
  model: ModelConstants = MODEL,
  weibull: WeibullFit,
  comps: SnapshotProtocol[],
): BoardRow[] {
  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => ({ p, s: scoreProtocol(p, snapshot.as_of, fitted, rates, model, weibull) }));
  scored.sort((a, b) => b.s.implied - a.s.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const enriched = scored.map(({ p, s }, i) => {
    const row: BoardRow = {
      rank: i + 1,
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      mcap_usd: p.mcap_usd,
      listed_at: p.listed_at,
      gecko_id: p.gecko_id,
      symbol: p.symbol,
      token_status: 'pre-token' as const,
      age_days: s.age,
      implied_value_usd: s.implied,
      launch_value_usd: Math.round(s.launchValue),
      launch_probability: s.probability,
      launch_readiness: 0,
      readiness_parts: { cat: 0, scale: 0, mom: 0, age: 0 },
      peers: [],
      why_rank: '',
      model: {
        category_multiple: s.categoryMultiple,
        category_iqr: s.categoryIqr,
        growth_multiplier: round4(s.growthMult),
        maturity_multiplier: round4(s.maturityMult),
        growth_pct: s.growthPct,
        age_days: s.age,
      },
      farm_steps: farmSteps(p, s.age),
    };
    return { row, p, s };
  });
  // Second pass: the readiness index, peers and why-line need the whole
  // board for context (the highest-rate category, the log-TVL range, the
  // median TVL), so they are computed after the ranks are fixed.
  const maxRate = Math.max(0.01, ...enriched.map(({ row }) => launchRate(row.category, rates)));
  const logTvls = enriched.map(({ row }) => Math.log10(Math.max(row.tvl_usd, 1)));
  const minLogTvl = Math.min(...logTvls);
  const maxLogTvl = Math.max(...logTvls);
  const medianTvl = median(enriched.map(({ row }) => row.tvl_usd));
  for (const { row, p, s } of enriched) {
    const { readiness, parts } = launchReadiness(
      p,
      { age: s.age, growthPct: s.growthPct },
      launchRate(p.category, rates),
      { maxRate, minLogTvl, maxLogTvl },
      model,
    );
    row.launch_readiness = readiness;
    row.readiness_parts = parts;
    row.peers = peersFor(p, comps);
    row.why_rank = whyRank(
      row,
      { categoryMultiple: s.categoryMultiple, growthPct: s.growthPct },
      { globalMultiple: fitted.global, globalRate: rates.global, medianTvl },
    );
  }
  return enriched.map((e) => e.row);
}

/** The live-token protocols, by market cap, for the dashboard's Live view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/**
 * The cross-cutting insights, all computed from the finished board at
 * run time: which categories' pre-token TVL is growing fastest, where the
 * pre-token TVL sits, the biggest launch values the odds are holding
 * back (outside the implied top 10), and how the 200 rows' launch odds
 * are distributed.
 */
export function buildInsights(board: BoardRow[]): RadarData['insights'] {
  const byCat = new Map<string, BoardRow[]>();
  for (const r of board) {
    const key = r.category ?? 'Unknown';
    const list = byCat.get(key) ?? [];
    list.push(r);
    byCat.set(key, list);
  }
  const category_momentum = [...byCat.entries()]
    .map(([category, rows]) => {
      const ch = rows.map((r) => r.change_7d).filter((c): c is number => typeof c === 'number');
      return { category, n: rows.length, median_change_7d: ch.length > 0 ? round4(median(ch)) : null };
    })
    .filter((x): x is { category: string; n: number; median_change_7d: number } => x.n >= 5 && x.median_change_7d !== null)
    .sort((a, b) => b.median_change_7d - a.median_change_7d);
  const pretoken_tvl = [...byCat.entries()]
    .map(([category, rows]) => ({ category, n: rows.length, tvl_usd: rows.reduce((s, r) => s + r.tvl_usd, 0) }))
    .sort((a, b) => b.tvl_usd - a.tvl_usd);
  const held_back = board
    .filter((r) => r.rank > 10)
    .sort((a, b) => b.launch_value_usd - a.launch_value_usd)
    .slice(0, 5)
    .map((r) => ({
      slug: r.slug,
      name: r.name,
      rank: r.rank,
      launch_value_usd: r.launch_value_usd,
      launch_probability: r.launch_probability,
      implied_value_usd: r.implied_value_usd,
    }));
  const odds_buckets: Array<{ label: string; count: number }> = [
    { label: '<5%', count: 0 },
    { label: '5–10%', count: 0 },
    { label: '10–20%', count: 0 },
    { label: '20–35%', count: 0 },
    { label: '35%+', count: 0 },
  ];
  for (const r of board) {
    const p = r.launch_probability;
    const i = p < 0.05 ? 0 : p < 0.1 ? 1 : p < 0.2 ? 2 : p < 0.35 ? 3 : 4;
    odds_buckets[i]!.count++;
  }
  const category_median_change: Record<string, number> = {};
  for (const [category, rows] of byCat) {
    const ch = rows.map((r) => r.change_7d).filter((c): c is number => typeof c === 'number');
    if (ch.length > 0) category_median_change[category] = round4(median(ch));
  }
  return { category_momentum, pretoken_tvl, held_back, odds_buckets, category_median_change };
}

// ---------------------------------------------------------------- held-out validation

/** The deterministic split bucket of a slug: first 8 hex digits of its sha256, mod 10. */
export function splitBucket(slug: string): number {
  const hex = createHash('sha256').update(slug, 'utf8').digest('hex').slice(0, 8);
  return parseInt(hex, 16) % 10;
}

function validationScore(name: string, preds: Array<{ pred: number; actual: number }>): ValidationScore {
  const errs = preds.map(({ pred, actual }) => Math.abs(pred - actual) / actual).sort((a, b) => a - b);
  const mid = Math.floor(errs.length / 2);
  const med = errs.length % 2 === 1 ? errs[mid]! : (errs[mid - 1]! + errs[mid]!) / 2;
  const within = preds.filter(({ pred, actual }) => pred / actual >= 0.5 && pred / actual <= 2).length;
  return {
    name,
    median_abs_pct_error: Math.round(med * 1000) / 10,
    within_2x_pct: Math.round((within / preds.length) * 1000) / 10,
    test_n: preds.length,
  };
}

/**
 * Held-out validation, computed from the snapshot itself: comparables split
 * by sha256(slug) mod 10 (train < 8, test >= 8), multiples fitted on train
 * only, predicted market cap = TVL x multiple, scored against the real
 * market cap. "Within 2x" means the prediction lands between 0.5x and 2x
 * of the actual market cap. Zero-market-cap entries cannot be scored and
 * are left out of the test set. The same three models are also scored on a
 * walk-forward split (train on protocols listed before a cutoff, test after
 * it), and the launch prior is scored as a classifier on the hash split.
 */
export function heldOutValidation(snapshot: Snapshot, model: ModelConstants = MODEL): HeldOutValidation {
  const comps = comparables(snapshot.protocols);
  const train = comps.filter((p) => splitBucket(p.slug) < 8);
  const test = comps.filter((p) => splitBucket(p.slug) >= 8 && (p.mcap_usd ?? 0) > 0);
  const fitted = fitMultiples(train, model);
  const catPreds = test.map((p) => ({ pred: p.tvl_usd * categoryMultiple(p.category, fitted), actual: p.mcap_usd! }));
  const flatPreds = test.map((p) => ({ pred: p.tvl_usd * 0.25, actual: p.mcap_usd! }));
  const globalPreds = test.map((p) => ({ pred: p.tvl_usd * fitted.global, actual: p.mcap_usd! }));
  return {
    split: 'sha256(slug) first 8 hex digits mod 10: train < 8, test >= 8',
    train_n: train.length,
    test_n: test.length,
    models: [
      validationScore('category median (shrunk, clamped)', catPreds),
      validationScore('flat 0.25 (starter)', flatPreds),
      validationScore('flat global median', globalPreds),
    ],
    walk_forward: walkForwardValidation(snapshot, model),
    prior: priorValidation(snapshot, model),
  };
}

/**
 * Walk-forward validation: for each cutoff date, fit the multiples on
 * comparables listed before the cutoff and score them on comparables listed
 * after it. A model that only works on a random split but not forward in
 * time is fitting noise; the walk-forward numbers are the honest second
 * opinion. Cutoffs are chosen so every test set holds at least 30
 * comparables.
 */
export function walkForwardValidation(snapshot: Snapshot, model: ModelConstants = MODEL): WalkForwardSplit[] {
  const comps = comparables(snapshot.protocols);
  const out: WalkForwardSplit[] = [];
  for (const cutoff of ['2024-01-01', '2024-07-01', '2025-01-01']) {
    const train = comps.filter((p) => p.listed_at !== null && p.listed_at < cutoff);
    const test = comps.filter((p) => p.listed_at !== null && p.listed_at >= cutoff && (p.mcap_usd ?? 0) > 0);
    if (test.length < 30) continue;
    const fitted = fitMultiples(train, model);
    out.push({
      cutoff,
      train_n: train.length,
      test_n: test.length,
      models: [
        validationScore(
          'category median (shrunk, clamped)',
          test.map((p) => ({ pred: p.tvl_usd * categoryMultiple(p.category, fitted), actual: p.mcap_usd! })),
        ),
        validationScore('flat 0.25 (starter)', test.map((p) => ({ pred: p.tvl_usd * 0.25, actual: p.mcap_usd! }))),
        validationScore('flat global median', test.map((p) => ({ pred: p.tvl_usd * fitted.global, actual: p.mcap_usd! }))),
      ],
    });
  }
  return out;
}

/**
 * The launch prior as a classifier: train each prior on the hash train
 * split, predict P(token) for the hash test split, and score Brier, log
 * loss and AUC against whether the protocol actually carries a token.
 * The v1 hand-set prior is scored as the baseline it replaced (its group
 * rates and size/age factors are written out here, not read from MODEL —
 * the model no longer carries them). AUC below 0.5 means the prior ranks
 * backwards: it gives the highest odds to the categories least likely to
 * have a token.
 */
export function priorValidation(snapshot: Snapshot, model: ModelConstants = MODEL): PriorScore[] {
  const trainRows = snapshot.protocols.filter((p) => splitBucket(p.slug) < 8);
  const testRows = snapshot.protocols.filter((p) => splitBucket(p.slug) >= 8);
  const rates = fitLaunchRates(trainRows, model);
  const weibull = fitWeibull(trainRows, snapshot.as_of, model);
  const v1 = (p: SnapshotProtocol): number => {
    const infra = new Set(['Chain', 'Bridge', 'Canonical Bridge', 'Cross Chain Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaking', 'Staking Pool']);
    const other = new Set(['CEX', 'Payments', 'Insurance', 'CeDeFi', 'Governance Incentives', 'Stablecoin Wrapper', 'Collateral Markets', 'Collateral Management', 'Dual-Token Stablecoin', 'Privacy']);
    const base = infra.has(p.category ?? '') ? 0.4 : other.has(p.category ?? '') ? 0.35 : 0.65;
    const tvlFactor = clamp(0.85 + (0.15 * (Math.log10(Math.max(p.tvl_usd, 1)) - 7)) / 2, 0.8, 1.15);
    const age = ageDays(snapshot.as_of, p.listed_at);
    const ageFactor = age === null ? 0.85 : age < 180 ? 1.0 : age < 365 ? 0.95 : age < 730 ? 0.9 : age < 1095 ? 0.8 : 0.7;
    return clamp(base * tvlFactor * ageFactor, 0.1, 0.9);
  };
  const scorePrior = (name: string, fn: (p: SnapshotProtocol) => number): PriorScore => {
    let brier = 0;
    let ll = 0;
    let pos = 0;
    const pairs: Array<[number, number]> = [];
    for (const p of testRows) {
      const y = p.mcap_usd !== null ? 1 : 0;
      if (y === 1) pos++;
      const q = clamp(fn(p), 0.01, 0.99);
      brier += (q - y) ** 2;
      ll += -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
      pairs.push([q, y]);
    }
    const positives = pairs.filter(([, y]) => y === 1).map(([q]) => q);
    const negatives = pairs.filter(([, y]) => y === 0).map(([q]) => q);
    let conc = 0;
    for (const a of positives) for (const b of negatives) conc += a > b ? 1 : a === b ? 0.5 : 0;
    const n = testRows.length;
    return {
      name,
      brier: Math.round((brier / n) * 10000) / 10000,
      log_loss: Math.round((ll / n) * 10000) / 10000,
      auc: Math.round((conc / (positives.length * negatives.length)) * 1000) / 1000,
      test_n: n,
      positives: pos,
    };
  };
  return [
    scorePrior('empirical category rate + Weibull age discount (v3)', (p) =>
      launchProbability(p, ageDays(snapshot.as_of, p.listed_at), rates, model, weibull)),
    scorePrior('empirical category rate + hand-set age steps (v2)', (p) =>
      launchProbabilitySteps(p, ageDays(snapshot.as_of, p.listed_at), rates, model)),
    scorePrior('empirical category rate, no age discount', (p) =>
      round4(clamp(launchRate(p.category, rates), model.prob_floor, model.prob_cap))),
    scorePrior('hand-set group prior (v1)', v1),
    scorePrior('global token rate (constant)', () => rates.global),
  ];
}

// ---------------------------------------------------------------- assembly

/** A board row's compact form for data.js. `onWatchlist` marks the 40 floor rows so the page can filter them. */
export function radarRow(row: BoardRow, onWatchlist: boolean): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    launch_value_usd: row.launch_value_usd,
    launch_probability: row.launch_probability,
    launch_readiness: row.launch_readiness,
    readiness_parts: row.readiness_parts,
    peers: row.peers,
    why_rank: row.why_rank,
    category_multiple: row.model.category_multiple,
    category_iqr: row.model.category_iqr,
    growth_multiplier: row.model.growth_multiplier,
    maturity_multiplier: row.model.maturity_multiplier,
    watchlist: onWatchlist,
    farm_steps: row.farm_steps,
  };
}

/** Everything the dashboard needs, from one snapshot. */
export function buildRadar(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const fitted = fitMultiples(snapshot.protocols, model);
  const rates = fitLaunchRates(snapshot.protocols, model);
  const weibull = fitWeibull(snapshot.protocols, snapshot.as_of, model);
  const comps = comparables(snapshot.protocols);
  const board = buildBoard(snapshot, fitted, rates, model, weibull, comps);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const categoryMultiples: Record<string, { n: number; multiple: number; iqr: [number, number] }> = {};
  for (const [cat, v] of fitted.byCategory) categoryMultiples[cat] = v;
  const launchRates: Record<string, { n: number; rate: number }> = {};
  for (const [cat, v] of rates.byCategory) launchRates[cat] = v;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    generated_at: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    global_multiple: fitted.global,
    category_multiples: categoryMultiples,
    global_launch_rate: rates.global,
    launch_rates: launchRates,
    weibull,
    validation: heldOutValidation(snapshot, model),
    insights: buildInsights(board),
    rows: board.map((r) => radarRow(r, snapshot.watchlist.includes(r.slug))),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) {
    throw new SnapshotError('snapshot.json has no floor_usd');
  }
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) {
    throw new SnapshotError('snapshot.json has no watchlist of slugs');
  }
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, flat values, farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  if (board.length < 100 || board.length > 4000) return `the board has ${board.length} rows, outside 100..4000`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) {
    if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  }
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
    if (!Number.isSafeInteger(r.launch_value_usd) || r.launch_value_usd <= 0) return `${r.slug} has launch value ${r.launch_value_usd}`;
    if (typeof r.launch_probability !== 'number' || r.launch_probability < 0.02 || r.launch_probability > 0.9) {
      return `${r.slug} has launch probability ${r.launch_probability}`;
    }
    if (typeof r.launch_readiness !== 'number' || r.launch_readiness < 0 || r.launch_readiness > 100) {
      return `${r.slug} has launch readiness ${r.launch_readiness}`;
    }
    if (r.peers.length !== 3) return `${r.slug} has ${r.peers.length} peers, not 3`;
    if (typeof r.why_rank !== 'string' || r.why_rank.length < 20) return `${r.slug} has no why_rank line`;
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    }
  }
  return null;
}

/** The dashboard's dataset as one script: no fetch, no network, the page reads window.RADAR. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
