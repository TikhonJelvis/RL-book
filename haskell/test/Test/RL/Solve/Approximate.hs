{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}
module Test.RL.Solve.Approximate where

import           Control.Monad.Bayes.Class                ( bernoulli )
import           Control.Monad.Bayes.Sampler              ( sampleIO )

import           Data.Bool                                ( bool )

import           Numeric.LinearAlgebra                    ( Container
                                                          , R
                                                          , scalar
                                                          )

import qualified Streaming.Prelude                       as Streaming

import           Text.Printf                              ( printf )

import qualified RL.Iterate                              as Iterate
import           RL.Matrix                                ( allWithin )
import           RL.Within                                ( within )

import qualified RL.Approx                               as Approx
import qualified RL.Approx.Linear                        as Linear
import qualified RL.Approx.Tabular                       as Tabular

import           RL.Process.Finite                        ( toRewardProcess )

import           RL.Solve.Approximate                     ( evaluateFiniteMRP
                                                          , evaluateMRP
                                                          )

import           Test.Assert                              ( assertWithin )
import           Test.Processes                           ( flipFlop )

import           Test.Tasty
import           Test.Tasty.HUnit

tests :: TestTree
tests = testGroup
  "Approximate"
  [ testGroup
    "evaluateFiniteMRP"
    [ testCase "flipFlop + Tabular" $ do
      let v₀     = Tabular.create @[] [True, False]
          vs     = evaluateFiniteMRP (flipFlop 0.7) 0.99 v₀
          Just v = Iterate.converge (within 1e-5) vs
      assertWithin 0.1 (Approx.eval' v [True, False]) (scalar 170)
    , testCase "flipFlop + Linear" $ do
      let v₀     = Linear.create 0 [bool 0 1, bool 1 0]
          vs     = evaluateFiniteMRP (flipFlop 0.7) 0.99 v₀
          Just v = Iterate.converge (within 1e-5) vs
      assertWithin 0.1 (Approx.eval' v [True, False]) (scalar 170)
    ]
  , testGroup
    "evaluateMRP"
    [ testCase "flipFlop + Tabular" $ do
        let v₀      = Tabular.create @[] [True, False]
            states  = bernoulli 0.5
            process = toRewardProcess $ flipFlop 0.7
            vs      = evaluateMRP process states 0.99 5 v₀

        Just v <- sampleIO $ Iterate.converge' (within 1e-4) vs
        assertWithin 0.1 (Approx.eval' v [True, False]) (scalar 170)
    ]
  ]
