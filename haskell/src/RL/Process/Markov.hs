{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Markov processes that may or may not include rewards.
module RL.Process.Markov where

import           Control.Monad.Bayes.Class                ( MonadSample(..) )
import           Control.Monad.Bayes.Sampler              ( SamplerIO )
import           Control.Monad.Trans                      ( MonadTrans
                                                          , lift
                                                          )
import           Control.Monad.Writer.Strict              ( MonadWriter
                                                          , WriterT
                                                          )
import qualified Control.Monad.Writer.Strict             as Writer

import           Data.Monoid                              ( Sum(..) )
import           Data.Vector.Storable                     ( Storable )

import qualified Numeric.LinearAlgebra                   as Matrix

import           Streaming                                ( Of
                                                          , Stream
                                                          )
import           Streaming.Prelude                        ( yield )
import qualified Streaming.Prelude                       as Streaming

-- * Markov Processes

-- ** Processes

-- | A Markov process for some (probability) monad @m@ is defined by
-- its transition function.
newtype MarkovProcess m s = MarkovProcess { step :: s -> m s }

-- ** Simulations

-- | A single trace of a Markov process, implemented as a stream of
-- states.
type Trace m s = Stream (Of s) m ()

-- | Run a single simulation trace of the given 'MarkovProcess',
-- returning each visited state (including the start state).
simulate :: Monad m
         => m s
         -- ^ A distribution of states to draw from to start the
         -- trace.
         -> MarkovProcess m s
         -- ^ The Markov process to simulate.
         -> Trace m s
simulate start MarkovProcess { step } = Streaming.iterateM step start

-- * Markov Reward Processes

-- ** Rewards

-- | A unitless reward that we can optimize against.
type Reward = Sum Double

-- | Gain (or lose) some reward.
earn :: MonadWriter Reward m => Double -> m ()
earn n = Writer.tell (Sum n)

-- | A monad transformer for earning 'Reward's.
newtype WithReward m a = WithReward (WriterT Reward m a)
  deriving newtype (Functor, Applicative, Monad, MonadTrans, MonadWriter Reward)

runWithReward :: WithReward m a -> m (a, Reward)
runWithReward (WithReward action) = Writer.runWriterT action

instance MonadSample m => MonadSample (WithReward m) where
  random      = lift random
  bernoulli   = lift . bernoulli
  categorical = lift . categorical

-- ** Reward Processes

-- | A 'MarkovProcess' that yields a potentially random reward for
-- each transition between two states.
type MarkovRewardProcess m s = MarkovProcess (WithReward m) s

-- | Step a 'MarkovRewardProcess', returning the new state and reward.
step' :: Monad m => MarkovRewardProcess m s -> s -> m (s, Double)
step' MarkovProcess { step } s = do
  (s', r) <- runWithReward (step s)
  pure (s', getSum r)

-- ** Simulations

-- | A transition from 'state' to 'next' in some Markov reward
-- process, giving an instaneous reward 'reward'.
data Step s = Step
  { state  :: !s
  , next   :: !s
  , reward :: !Reward
  }
  deriving (Show, Eq)

-- | Run a simulation trace of the given reward process, yielding the
-- state transition and *instantaneous* reward for each step.
simulateReward :: Monad m
               => MarkovRewardProcess m s
               -- ^ The reward process to simulate.
               -> m s
               -- ^ A distribution of states to draw from to start the
               -- trace.
               -> Trace m (Step s)
simulateReward MarkovProcess { step } start = lift start >>= go
 where
  go !state = do
    (next, reward) <- lift $ runWithReward (step state)
    yield Step { state, next, reward }
    go next
