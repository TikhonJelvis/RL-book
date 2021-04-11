{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
-- | Algorithms for "solving" finite Markov reward processes with
-- dynamic programming.
module RL.Solve.Dynamic where

import qualified Data.Vector.Storable                    as Vector

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , R
                                                          , scalar
                                                          , scale
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Process.Finite                        ( FiniteMarkovProcess(..)
                                                          , FiniteMarkovRewardProcess(..)
                                                          , fromRewardProcess
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
