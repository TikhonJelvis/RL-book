module Test.RL.Solve.Dynamic where

import           Control.Monad.Bayes.Class                ( bernoulli )

import           Numeric.LinearAlgebra                    ( scalar )

import qualified RL.Iterate                              as Iterate
import           RL.Matrix                                ( allWithin )

import           RL.Process.Finite
import           RL.Process.Markov                        ( MarkovProcess(..)
                                                          , earn
                                                          )

import           RL.Solve.Dynamic                         ( evaluateMRP )

import           Test.Processes                           ( flipFlop )

import           Test.Tasty
import           Test.Tasty.HUnit

tests :: TestTree
tests = testGroup
  "Dynamic"
  [ testGroup
      "evaluateMRP"
      [ testCase "flipFlop" $ do
          let vs     = evaluateMRP (flipFlop 0.7) 0.99
              Just v = Iterate.converge (allWithin 1e-5) vs
          assertBool "Not within 0.001 of <170, 170>."
            $ allWithin 1e-3 v (scalar 170)
      ]
  ]
