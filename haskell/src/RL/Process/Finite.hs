{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE ViewPatterns #-}
module RL.Process.Finite (FiniteMarkovProcess, states, toMarkov) where


import qualified Control.Monad.Bayes.Class               as Bayes
import           Control.Monad.Bayes.Class                ( MonadSample )
import           Control.Monad.Bayes.Enumerator           ( Enumerator )
import qualified Control.Monad.Bayes.Enumerator          as Enumerator

import           Data.HashMap.Strict                      ( HashMap )
import qualified Data.HashMap.Strict                     as HashMap
import           Data.Hashable                            ( Hashable )
import qualified Data.List                               as List
import           Data.Vector                              ( Vector )
import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (!)
                                                          , Matrix
                                                          , R
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Process.Markov                        ( MarkovProcess(..) )

-- * Finite Markov Processes

-- | A Markov process with a finite number of states which can be
-- recorded as a transition matrix.
data FiniteMarkovProcess s = FiniteMarkovProcess
  { stateIndices :: HashMap s Int
    -- ^ The index of every state in the transition matrix.
  , states       :: Vector s
    -- ^ States, in an order that corresponds to their indices in
    -- `stateIndices`.
  , transition   :: Matrix R
    -- ^ For a N state process, this is an N × N matrix where each
    -- cell represents the probability of moving from one state to
    -- another.
  }

-- | The normal 'MarkovProcess' that corresponds to the given finite
-- process. This lets us do stuff like run simulations and other
-- general-purpose algorithms over finite processes.
--
-- The transition function of the resulting `MarkovProcess` will throw
-- an error if given a state that is not in the `FiniteMarkovProcess`.
toMarkov :: (Hashable s, Eq s, MonadSample m)
         => FiniteMarkovProcess s
         -> MarkovProcess m s
toMarkov FiniteMarkovProcess { states, stateIndices, transition } =
  MarkovProcess $ \s -> case HashMap.lookup s stateIndices of
    Just i -> do
      j <- Bayes.categorical (transition ! i)
      pure (states V.! j)
    Nothing ->
      error "Invalid state passed to process created from a finite Markov process."

-- | Create a finite process from the given Markov process and its
-- full set of states.
--
-- The resulting finite process may not be valid if the given set of
-- states does not cover all the valid states of the given process.
fromMarkov :: (Hashable s, Eq s, Ord s)
           => [s]
           -> MarkovProcess Enumerator s
           -> FiniteMarkovProcess s
fromMarkov (List.sort -> states) MarkovProcess { step } = FiniteMarkovProcess
  { stateIndices = HashMap.fromList [ (s, i) | s <- states | i <- [0..] ]
  , states       = V.fromList states
  , transition   = Matrix.fromRows $ toRow <$> states
  }
  where toRow s = Matrix.fromList [ p | (_, p) <- Enumerator.enumerate (step s) ]
