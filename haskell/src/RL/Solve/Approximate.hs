{-# LANGUAGE NamedFieldPuns #-}
module RL.Solve.Approximate where

import           Control.Monad                            ( replicateM )

import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , scale
                                                          )

import           Streaming                                ( Of
                                                          , Stream
                                                          )
import qualified Streaming.Prelude                       as Streaming

import           RL.FunctionApproximation                 ( Approx )
import qualified RL.FunctionApproximation                as Approx

import           RL.Process.Finite                        ( FiniteMarkovProcess(..)
                                                          , FiniteMarkovRewardProcess(..)
                                                          )
import           RL.Process.Markov                        ( MarkovRewardProcess )

evaluateFiniteMRP :: Approx v
                  => FiniteMarkovRewardProcess s
                  -> Double
                  -- ^ Discount factor (γ)
                  -> v s
                  -- ^ Starting value function approximation (V₀)
                  -> [v s]
evaluateFiniteMRP FiniteMarkovRewardProcess { process, expectedRewards } γ v₀ = iterate
  update
  v₀
 where
  update v = Approx.update v (states process) updated
   where
    updated = expectedRewards + scale γ (transition process #> vs)
    vs      = Approx.eval' v (states process)

evaluateMRP :: (Approx v, Monad m)
            => MarkovRewardProcess m s
            -> m s
            -- ^ Distribution of start states.
            -> Double
            -- ^ Discount factor (γ)
            -> Int
            -- ^ Number of states to sample at each step.
            -> v s
            -- ^ Starting value function approximation (V₀)
            -> Stream (Of (v s)) m ()
evaluateMRP process startStates γ n v₀ = Streaming.iterateM update (pure v₀)
 where
  update v = do
    states <- replicateM n startStates
    let return (s, r) = r + γ * Approx.eval v s
    pure $ Approx.update v (V.fromList states) undefined
