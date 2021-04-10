{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE ViewPatterns #-}
module RL.Process.Finite
  ( FiniteMarkovProcess(..)
  , toProcess
  , fromProcess
  , FiniteMarkovRewardProcess(..)
  , toRewardProcess
  , fromRewardProcess
  ) where


import qualified Control.Monad.Bayes.Class               as Bayes
import           Control.Monad.Bayes.Class                ( MonadSample )
import           Control.Monad.Bayes.Enumerator           ( Enumerator )
import qualified Control.Monad.Bayes.Enumerator          as Enumerator

import           Data.HashMap.Strict                      ( HashMap )
import qualified Data.HashMap.Strict                     as HashMap
import           Data.Hashable                            ( Hashable )
import qualified Data.List                               as List
import           Data.Monoid                              ( Sum(..) )
import           Data.Vector                              ( Vector )
import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (!)
                                                          , Matrix
                                                          , R
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Matrix                                ( sumRows )
import           RL.Process.Markov                        ( MarkovProcess(..)
                                                          , MarkovRewardProcess(..)
                                                          , Reward
                                                          , earn
                                                          , runWithReward
                                                          , step'
                                                          )

-- * Finite Markov Processes

-- | A Markov process with a finite number of states which can be
-- recorded as a transition matrix.
data FiniteMarkovProcess s = FiniteMarkovProcess
  { stateIndices :: !(HashMap s Int)
    -- ^ The index of every state in the transition matrix.
  , states       :: !(Vector s)
    -- ^ States, in an order that corresponds to their indices in
    -- `stateIndices`.
  , transition   :: !(Matrix R)
    -- ^ For a N state process, this is an N × N matrix where each
    -- cell represents the probability of moving from one state to
    -- another.
  }
  deriving (Show, Eq)

-- | The normal 'MarkovProcess' that corresponds to the given finite
-- process. This lets us do stuff like run simulations and other
-- general-purpose algorithms over finite processes.
--
-- The transition function of the resulting 'MarkovProcess' will throw
-- an error if given a state that is not in the 'FiniteMarkovProcess'.
toProcess :: (Hashable s, Eq s, MonadSample m)
          => FiniteMarkovProcess s
          -> MarkovProcess m s
toProcess FiniteMarkovProcess { states, stateIndices, transition } =
  MarkovProcess $ \s -> case HashMap.lookup s stateIndices of
    Just i -> do
      j <- Bayes.categorical (transition ! i)
      pure (states V.! j)
    Nothing ->
      error "Invalid state passed to process created from a finite Markov process."

-- | Create a finite process from the given Markov process and its
-- full set of states.
--
-- The resulting finite process may not be valid if the list of states
-- does not cover all the valid states of the given process.
fromProcess :: (Hashable s, Eq s, Ord s)
            => [s]
            -- ^ All the states in the process.
            -> MarkovProcess Enumerator s
            -- ^ The underlying process.
            -> FiniteMarkovProcess s
fromProcess (List.sort -> states) MarkovProcess { step } = FiniteMarkovProcess
  { stateIndices = HashMap.fromList [ (s, i) | s <- states | i <- [0..] ]
  , states       = V.fromList states
  , transition   = Matrix.fromRows $ toRow <$> states
  }
  where toRow s = Matrix.fromList [ p | (_, p) <- Enumerator.enumerate (step s) ]


-- * Finite Markov Reward Processes

-- | A Markov reward process with a finite number of states,
-- represented as a 'FiniteMarkovProcess' along with a reward matrix.
data FiniteMarkovRewardProcess s = FiniteMarkovRewardProcess
  { process         :: !(FiniteMarkovProcess s)
  , rewards         :: !(Matrix R)
  , expectedRewards :: !(Matrix.Vector R)
  }
  deriving (Show, Eq)

-- | The normal 'MarkovRewardProcess' that corresponds to the given
-- finite process. This lets us do stuff like run simulations and other
-- general-purpose algorithms over finite processes.
--
-- The transition function of the resutling 'MarkovRewardProcess' will
-- throw an error if given a state that is not in the
-- 'FiniteMarkovRewardProcess'.
toRewardProcess :: (Hashable s, Eq s, MonadSample m)
                => FiniteMarkovRewardProcess s
                -> MarkovRewardProcess m s
toRewardProcess FiniteMarkovRewardProcess { process, rewards } = MarkovProcess $ \s ->
  let FiniteMarkovProcess { stateIndices, states, transition } = process
  in  case HashMap.lookup s stateIndices of
        Just i -> do
          j <- Bayes.categorical (transition ! i)
          earn (rewards ! i ! j)
          pure (states V.! j)
        Nothing ->
          error
            "Invalid state passed to process created from a finite Markov reward process."

-- | Create a finite reward process from the given Markov process and
-- its full set of states.
--
-- The resulting finite process may not be valid if the list of states
-- does not cover all the valid states of the given process.
fromRewardProcess :: (Hashable s, Eq s, Ord s)
                  => [s]
                  -- ^ All the states in the process.
                  -> MarkovRewardProcess Enumerator s
                  -- ^ The underlying process.
                  -> FiniteMarkovRewardProcess s
fromRewardProcess (List.sort -> states) rewardProcess = FiniteMarkovRewardProcess
  { process
  , rewards
  , expectedRewards = sumRows $ transition * rewards
  }
 where
  rewards    = Matrix.fromLists $ map (getSum . fst) <$> results
  transition = Matrix.fromLists $ map snd <$> results

  results =
    [ [ (r, p) | ((_, r), p) <- Enumerator.enumerate (step' rewardProcess s) ]
    | s <- states
    ]

  process = FiniteMarkovProcess
    { stateIndices = HashMap.fromList [ (s, i) | s <- states | i <- [0..] ]
    , states       = V.fromList states
    , transition
    }
