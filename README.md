# multi-armed-bandits
Implementation of an example algorithm solving multi-armed bandits with gaussian rewards problem.

Assumes we know that each bandit has variance of rewards equal to 1.

Initially each bandit is drawn once. The order of subsequent draws is determined by finding bandit with the highest value of mean reward + 3 * its standard deviation.

## Usage

```clojure
(simulate bandits-distribution-standard-deviation number-of-bandits number-of-steps)
```

Returns vector consisting of:
  - collection of pairs [mean variance] defining each bandit,
  - vector of pairs [mean variance-of-mean] describing our knowledge of bandits based on drawn numbers,
  - vector of intigers - total numbers of draws from each bandit.
