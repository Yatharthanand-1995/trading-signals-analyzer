# 🔄 Trading Logic Flowchart

## Signal Generation Process Flow

```
┌─────────────────────┐
│   Start Analysis    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Fetch Market Data  │
│  - Current Price    │
│  - Volume           │
│  - Historical Data  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calculate Technical │
│    Indicators       │
├─────────────────────┤
│ • Moving Averages   │
│ • RSI               │
│ • MACD              │
│ • Bollinger Bands   │
│ • Support/Resist    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Score Components   │
├─────────────────────┤
│ Trend Score (40%)   │
│ Momentum (30%)      │
│ Volume (15%)        │
│ Volatility (15%)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calculate Total     │
│      Score          │
│    (0-100)          │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │ Score > 80? │──Yes──→ [STRONG_BUY]
    └──────┬──────┘
           │ No
           ▼
    ┌──────┴──────┐
    │ Score > 60? │──Yes──→ [BUY]
    └──────┬──────┘
           │ No
           ▼
    ┌──────┴──────┐
    │ Score > 40? │──Yes──→ [HOLD]
    └──────┬──────┘
           │ No
           ▼
    ┌──────┴──────┐
    │ Score > 20? │──Yes──→ [SELL]
    └──────┬──────┘
           │ No
           ▼
      [STRONG_SELL]
```

## Detailed Component Scoring

### 1. Trend Analysis (40% Weight)

```
┌─────────────────────────┐
│     Trend Analysis      │
├─────────────────────────┤
│ Price > SMA200? (+10)   │
│ Price > SMA50?  (+10)   │
│ Price > SMA20?  (+10)   │
│ MACD > Signal?  (+10)   │
└─────────────────────────┘
         │
         ▼
   Max Points: 40
```

### 2. Momentum Analysis (30% Weight)

```
┌─────────────────────────┐
│   Momentum Analysis     │
├─────────────────────────┤
│ RSI (30-70)?    (+15)   │
│ RSI Direction?  (+10)   │
│ Stochastic OK?  (+5)    │
└─────────────────────────┘
         │
         ▼
   Max Points: 30
```

### 3. Volume Analysis (15% Weight)

```
┌─────────────────────────┐
│    Volume Analysis      │
├─────────────────────────┤
│ Vol > Average?  (+10)   │
│ Vol Trend Up?   (+5)    │
└─────────────────────────┘
         │
         ▼
   Max Points: 15
```

### 4. Volatility Analysis (15% Weight)

```
┌─────────────────────────┐
│  Volatility Analysis    │
├─────────────────────────┤
│ BB Position OK? (+10)   │
│ ATR Stable?     (+5)    │
└─────────────────────────┘
         │
         ▼
   Max Points: 15
```

## Risk Management Flow

```
┌─────────────────────┐
│   Entry Signal      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calculate ATR       │
│ (14-period)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Set Stop Loss       │
│ Entry - (2 × ATR)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calculate Risk      │
│ Entry - Stop Loss   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Set Take Profits    │
│ TP1 = Entry + 2×Risk│
│ TP2 = Entry + 4×Risk│
│ TP3 = Entry + 6×Risk│
└─────────────────────┘
```

## Position Sizing Flow

```
┌─────────────────────┐
│ Account Size: $10K  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Risk Per Trade: 2%  │
│ ($200)              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Risk Per Share =    │
│ Entry - Stop Loss   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Shares to Buy =     │
│ $200 / Risk Per     │
│      Share          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Position   │
│ Size < Account?     │
└─────────────────────┘
```

## Data Validation Flow

```
┌─────────────────────┐
│  Fetch Data         │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │ Data Valid? │
    └──────┬──────┘
      No │ │ Yes
         │ └────────→ [Continue Analysis]
         ▼
┌─────────────────────┐
│ Retry (Max 3 times) │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │  Success?    │
    └──────┬──────┘
      No │ │ Yes
         │ └────────→ [Continue Analysis]
         ▼
   [Use Cached Data
    or Exit]
```

## Complete Trading Decision Tree

```
                          [Market Data]
                               │
                    ┌──────────┴──────────┐
                    │   Calculate Score   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
            Score > 60                  Score ≤ 60
                 │                           │
         ┌───────▼───────┐           ┌───────▼───────┐
         │   BUY ZONE    │           │  SELL ZONE    │
         └───────┬───────┘           └───────┬───────┘
                 │                           │
      ┌──────────┴──────────┐     ┌──────────┴──────────┐
      │                     │     │                     │
  Score > 80            Score ≤ 80  Score < 20      Score ≥ 20
      │                     │         │                 │
[STRONG_BUY]             [BUY]   [STRONG_SELL]      [SELL]
      │                     │         │                 │
      └──────────┬──────────┘         └────────┬────────┘
                 │                             │
                 ▼                             ▼
         [Set Entry/Exit]              [Set Entry/Exit
          for LONG]                     for SHORT]
```

## Error Handling Decision Tree

```
┌─────────────────────┐
│     API Call        │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  Success?   │
    └──────┬──────┘
      No │ │ Yes
         │ └────────→ [Process Data]
         ▼
┌─────────────────────┐
│  Network Error?     │─Yes─→ [Retry with Backoff]
└──────────┬──────────┘
           │ No
           ▼
┌─────────────────────┐
│  Rate Limited?      │─Yes─→ [Wait and Retry]
└──────────┬──────────┘
           │ No
           ▼
┌─────────────────────┐
│  Invalid Data?      │─Yes─→ [Use Cached Data]
└──────────┬──────────┘
           │ No
           ▼
    [Log Error & Exit]
```

## Signal Confirmation Matrix

| Indicator | Bullish | Neutral | Bearish |
|-----------|---------|---------|---------|
| Price vs MA | Above all | Mixed | Below all |
| RSI | 30-50 | 50-70 | >70 or <30 |
| MACD | Above signal | Crossing | Below signal |
| Volume | Above avg | Average | Below avg |
| Bollinger | Near lower | Middle | Near upper |

## Final Signal Logic

```python
if trend_score >= 30 and momentum_score >= 20:
    if total_score >= 80:
        signal = "STRONG_BUY"
    elif total_score >= 60:
        signal = "BUY"
elif trend_score <= 10 and momentum_score <= 10:
    if total_score <= 20:
        signal = "STRONG_SELL"
    elif total_score <= 40:
        signal = "SELL"
else:
    signal = "HOLD"
```