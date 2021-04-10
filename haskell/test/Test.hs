module Main where

import           Test.Tasty

import qualified Test.RL.Iterate                         as Iterate
import qualified Test.RL.Matrix                          as Matrix

import qualified Test.RL.Process.DynamicProgramming      as DynamicProgramming

tests :: TestTree
tests = testGroup
  "RL"
  [Iterate.tests, Matrix.tests, testGroup "Process" [DynamicProgramming.tests]]

main :: IO ()
main = defaultMain tests
