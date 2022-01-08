module Main where

import           Test.Tasty

import qualified Test.RL.Iterate                         as Iterate
import qualified Test.RL.Matrix                          as Matrix
import qualified Test.RL.Probability                     as Probability

import qualified Test.RL.Solve.Approximate               as Approximate
import qualified Test.RL.Solve.Dynamic                   as Dynamic

tests :: TestTree
tests = testGroup
  "RL"
  [ Iterate.tests
  , Matrix.tests
  , Probability.tests
  , testGroup "Solve" [
      Approximate.tests
      , Dynamic.tests
      ]
  ]

main :: IO ()
main = defaultMain tests
