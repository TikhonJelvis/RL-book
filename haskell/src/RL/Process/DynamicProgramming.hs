{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
-- | Algorithms for "solving" finite Markov reward processes with
-- dynamic programming.
module RL.Process.DynamicProgramming where

import           Control.Monad.Bayes.Class                ( bernoulli )

import qualified Data.Vector.Storable                    as Vector

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , R
                                                          , scale
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Process.Finite                        ( FiniteMarkovProcess(..)
                                                          , FiniteMarkovRewardProcess(..)
                                                          , fromRewardProcess
                                                          )
import           RL.Process.Markov                        ( MarkovProcess(..)
                                                          , earn
                                                          )

-- | Value functions can be represented as vectors with one value per
-- state.
type V = Matrix.Vector R

-- | Iteratively approximate the value function for the given Markov
-- reward process.
evaluateMRP :: FiniteMarkovRewardProcess s
            -> Double
            -- ^ Discount factor γ
            -> [V]
evaluateMRP FiniteMarkovRewardProcess {..} γ = iterate update v₀
 where
  update v = expectedRewards + scale γ (transition process #> v)
  v₀ = Vector.replicate (length $ states process) 0


-- TODO: organize test suite properly...
flipFlop :: Double -> FiniteMarkovRewardProcess Bool
flipFlop p = fromRewardProcess [True, False] $ MarkovProcess $ \s -> do
  next <- bernoulli (if s then 1 - p else p)
  earn $ if next == s then 1 else 2
  pure next
