{-# LANGUAGE NamedFieldPuns #-}
module RL.Solve.Approximate where

import           Control.Monad                            ( replicateM )
import           Control.Monad.Bayes.Class                ( MonadSample )

import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , scale
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           Streaming                                ( Of
                                                          , Stream
                                                          )
import qualified Streaming.Prelude                       as Streaming

import           RL.Approx                                ( Approx )
import qualified RL.Approx                               as Approx
import qualified RL.Probability                          as Probability

import           RL.Process.Finite                        ( FiniteMarkovProcess(..)
                                                          , FiniteMarkovRewardProcess(..)
                                                          )
import           RL.Process.Markov                        ( MarkovRewardProcess
                                                          , runWithReward
                                                          , step'
                                                          )

evaluateFiniteMRP :: Approx v s
                  => FiniteMarkovRewardProcess s
                  -> Double
                  -- ^ Discount factor (γ)
                  -> v s
                  -- ^ Starting value function approximation (V₀)
                  -> [v s]
evaluateFiniteMRP FiniteMarkovRewardProcess { process, expectedRewards } γ v₀ =
  iterate update v₀
 where
  update v = Approx.update v (states process) updated
   where
    updated = expectedRewards + scale γ (transition process #> vs)
    vs      = Approx.eval' v (states process)

evaluateMRP :: (Approx v s, Monad m, MonadSample m)
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
    rewards <- Matrix.fromList <$> mapM (expected return) states
    pure $ Approx.update v (V.fromList states) rewards

  expected f = Probability.expected n f . step' process
