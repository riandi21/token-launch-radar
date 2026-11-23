# token-launch-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/riandi21/token-launch-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/riandi21/token-launch-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Launch Radar ranks 200 pre-token protocols by implied launch value: TVL times a
category multiple fitted on launched tokens, times growth, maturity and the odds a token
ever lists. The board covers all 40 watchlist protocols plus 160 early ones below the
floor, each with a costed six-step farm plan, and a separate live view of the 24 largest
protocols that already trade. The model is fitted from the sealed snapshot at run time
and tested held-out — no protocol name is ever special-cased.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input token-radar-01-kit/input/snapshot.json
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=43vdbyh5&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
