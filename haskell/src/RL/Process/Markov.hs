{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Markov processes that may or may not include rewards.
module RL.Process.Markov where

import           Control.Monad.Bayes.Sampler              ( SamplerIO )
import           Control.Monad.Trans                      ( lift )
import           Control.Monad.Writer                     ( WriterT )
import qualified Control.Monad.Writer                    as Writer

import           Data.Monoid                              ( Sum )

import           Streaming                                ( Of
                                                          , Stream
                                                          )
import           Streaming.Prelude                        ( yield )
import qualified Streaming.Prelude                       as Streaming

-- * Markov Processes

-- | A Markov process for some (probability) monad @m@ is defined by
-- its transition function.
newtype MarkovProcess m s = MarkovProcess { step :: s -> m s }

type Trace m s = Stream (Of s) m ()

simulate :: Monad m => m s -> MarkovProcess m s -> Trace m s
simulate start MarkovProcess { step } = Streaming.iterateM step start

-- * Markov Reward Processes

type Reward = Sum Double

type MarkovRewardProcess m s = MarkovProcess (WriterT Reward m) s

data Step s = Step
  { state  :: !s
  , next   :: !s
  , reward :: !Reward
  }
  deriving (Show, Eq)

simulateReward :: Monad m => m s -> MarkovRewardProcess m s -> Trace m (Step s)
simulateReward start MarkovProcess { step } = lift start >>= go
 where
  go !state = do
    (next, reward) <- lift $ Writer.runWriterT (step state)
    yield Step { state, next, reward }
    go next
