module Test.RL.Iterate where

import           RL.Iterate                               ( converge )

import           Test.Tasty
import           Test.Tasty.HUnit

tests :: TestTree
tests = testGroup "Iterate" [test_converge]

test_converge :: TestTree
test_converge = testCase "converge" $ do
  converge (\a b -> a / b > 0.8) [0 ..] @?= Just 5.0
  converge (/=) "aabc" @?= Just 'a'
  converge (==) [1.0] @?= Nothing
