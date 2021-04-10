module Test.RL.Process.DynamicProgramming where

import           Control.Monad.Bayes.Class                ( bernoulli )

import           Numeric.LinearAlgebra                    ( scalar )

import qualified RL.Iterate                              as Iterate
import           RL.Matrix                                ( allWithin )

import           RL.Process.DynamicProgramming            ( evaluateMRP )
import           RL.Process.Finite
import           RL.Process.Markov                        ( MarkovProcess(..)
                                                          , earn
                                                          )

import           Test.Tasty
import           Test.Tasty.HUnit

tests :: TestTree
tests = testGroup "DynamicProgramming" [test_evaluateMRP]

test_evaluateMRP :: TestTree
test_evaluateMRP = testGroup
  "evaluateMRP"
  [ testCase "flipFlop" $ do
      let Just result =
            Iterate.converge (allWithin 1e-5) (evaluateMRP (flipFlop 0.7) 0.99)
      assertBool "Within 0.001 of <170, 170>." $ allWithin 1e-3 result (scalar 170)
  ]
 where
  flipFlop :: Double -> FiniteMarkovRewardProcess Bool
  flipFlop p = fromRewardProcess [True, False] $ MarkovProcess $ \s -> do
    next <- bernoulli (if s then 1 - p else p)
    earn $ if next == s then 1 else 2
    pure next

