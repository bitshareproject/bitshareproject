## Abstract

What the Code Does
A hierarchical Bayesian model for estimating a trader's expected value (EV) on any given FX trade, conditioned on the setup they're taking and the market context.

Pipeline:

Pulls historical FX candles from OANDA across a basket of pairs and multiple timeframes (H1, H4, D).

Detects boxes (ranges) on every bar and tags it as inside, breakout_up, breakout_dn, fakeout_up, or fakeout_dn, along with box features: height in ATR, breakout distance, VWAP distance/slope, position in box, touch balance.

Computes basket strength per currency: how strong each base/quote currency is relative to the whole FX basket, z-scored and sign-aligned to the trade direction. Positive = tailwind.

Pulls an economic calendar and flags any trade taken within ±N minutes of a high-impact release, tagging which side of the pair the release hits.

Matches each historical trade to the market state at its entry: box state, regime, session, time bracket, vol bucket, event context, basket strength, timeframe.

Computes a soft setup distribution per trade — how much each entry looks like range, breakout, reversal, or event — using box state, direction, event flag, and the continuous box/strength features.

Builds four separate hierarchical Bayesian libraries — one per setup. Each library pools trader performance down a chain of cells from finest to coarsest:

text
trader × pair × regime × session × vol × box_state × event_ccy
  × base_strength × quote_strength
→ trader × pair × regime × event_ccy × strengths
→ trader × pair × event_ccy × strengths
→ trader × regime × event_ccy × strengths
→ trader × time_bracket × event_ccy × base_strength
→ trader × event_ccy × base_strength
→ trader × base_strength
→ trader
→ style × time_bracket × regime × base_strength
→ style × regime
→ pair × regime × event_ccy × strengths
→ pair × event_ccy × base_strength
→ pair × base_strength
→ base_ccy × event_ccy × base_strength
→ quote_ccy × event_ccy × quote_strength
→ base_ccy | quote_ccy
→ regime × event_ccy
→ event_ccy
→ time_bracket | regime
→ base_strength_bucket | quote_strength_bucket
→ global
Every cell holds a time-decayed P(win), P(loss), P(scratch), EV_bps, avg_win, avg_loss, payoff_ratio, profit_factor, and cluster-inflated Wilson bounds.

Predicts EV for a new trade by:

Computing its soft setup distribution.

Looking up P(win | setup, context) and EV_bps(setup, context) for the trader under that regime, pair, currency strength, and event context, in each setup's library.

Combining: p_win_final = Σ p_setup · p_win_setup and ev_bps_final = Σ p_setup · ev_bps_setup.

So the final number reflects both how likely the trade is to be each type of setup and the trader's historical performance under that setup in that exact context — basket strength, regime, session, vol, event proximity, timeframe, and hold time all included.

Why Hierarchical Bayesian Pooling Beats ML for This Problem
1. Data per trader per setup per context is tiny.
A trader has maybe 40–200 trades. Split across four setups, five pairs, four regimes, six time brackets, three vol buckets, three base-strength buckets, three quote-strength buckets, two event currencies — the finest cells have zero to three observations. ML either overfits those cells or collapses the trader into the population mean. Hierarchical Bayes shrinks each cell toward its parent in proportion to how much data it has, so sparse cells borrow strength from the trader, pair, style, currency, and global priors.

2. Trades are genuinely nested, not i.i.d.
They cluster by trader → pair → currency → regime → session → timeframe → event → strength. That's the structure of the data. Hierarchical Bayes imposes it as a prior. ML has to learn it from scratch — and can't when the finest cells are empty.

3. Every prediction comes with honest uncertainty.
Each cell carries an effective sample size and cluster-inflated Wilson bounds. A trader with 15 trades gets a wide interval and heavy shrinkage. A trader with 2,000 trades gets a narrow one. ML gives a point estimate with no error bar and no visibility into how much to trust it.

4. It separates skill from luck.
Trades cluster by trader-day and by event, so a naive ML model will happily learn "T007 is great at NFP" from three lucky trades. Hierarchical shrinkage means three trades barely move the trader's estimate, and cluster inflation widens the interval for within-day autocorrelation.

5. It handles regime change gracefully.
When the market shifts and fine cells go empty, the chain naturally falls back to coarser, still-populated cells and widens the interval. ML sees distribution shift, degrades silently, and needs retraining.

6. It respects time without retraining.
Every observation is decayed by exp(-ln(2)·age/halflife). Recent trades dominate the posterior automatically — no fixed window, no retraining cycle.

7. It's auditable.
Any prediction decomposes: "60% global prior, 25% pair, 10% trader, 5% trader × pair × regime × base_strength." You can point at the cell that moved the number and defend it in a risk meeting. ML gives an opaque score.

8. It's a query-able posterior, not a frozen model.
Once built, the library answers any conditional question — "T042's reversal EV on H4 in London during a vol regime with a strong base and weak quote?" — without refitting. ML would need to be retrained per question.

9. It treats each setup as its own problem.
Event trades have different variance than range trades. A trader good at range can be bad at events. Four independent chains, each pooled within its own setup family — no cross-contamination.

10. It's calibrated by construction.
The posterior mean is the Bayes-optimal estimator under the prior. Wilson bounds add frequentist coverage. You get a Bayesian posterior and a frequentist guarantee.

The core insight: trader EV estimation is sparse, nested, time-varying, and multi-dimensional. That's exactly the regime hierarchical Bayes was designed for. ML assumes dense i.i.d. samples — the opposite of what you have. If you ever want to layer ML on top, the natural move is to feed the pooled Bayesian estimate in as a feature, not to replace it.
