module Test.RL.Probability where

import           Control.Monad.Bayes.Class
import           Control.Monad.Bayes.Sampler

import           Data.Bool                                ( bool )

import           Text.Printf                              ( printf )

import           RL.Probability
import           RL.Within                                ( within )

import           Test.Assert                              ( assertWithin )

import           Test.Tasty
import           Test.Tasty.HUnit
import           Test.Tasty.QuickCheck             hiding ( within )

tests :: TestTree
tests = testGroup
  "Probability"
  [ testGroup
      "expected"
      [ testCase "bernoulli" $ do
        x <- sampleIO $ expected 1000 (bool 0 1) (bernoulli 0.5)
        assertWithin 0.05 x 0.5
      , testProperty "poisson" $ forAll (choose (0.01, 5)) $ \λ ->
        ioProperty $ do
          x <- sampleIO $ expected 4000 fromIntegral (poisson λ)
          assertWithin 0.2 x λ
      ]
  ]
