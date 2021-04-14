-- | Various Markov processes that we can reuse across multiple test
-- modules.
module Test.Processes where

import           Control.Monad.Bayes.Class                ( bernoulli )

import           RL.Process.Finite
import           RL.Process.Markov                        ( MarkovProcess(..)
                                                          , earn
                                                          )


flipFlop :: Double -> FiniteMarkovRewardProcess Bool
flipFlop p = fromRewardProcess [True, False] $ MarkovProcess $ \s -> do
  next <- bernoulli (if s then 1 - p else p)
  earn $ if next == s then 1 else 2
  pure next
