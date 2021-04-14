{-# LANGUAGE OverloadedLists #-}
module Test.RL.Solve.Approximate where

import           Control.Monad.Bayes.Class                ( bernoulli )
import           Control.Monad.Bayes.Sampler              ( sampleIO )

import           Data.Bool                                ( bool )

import           Numeric.LinearAlgebra                    ( scalar )

import qualified Streaming.Prelude                       as Streaming

import           Text.Printf                              ( printf )

import qualified RL.FunctionApproximation                as Approx
import qualified RL.Iterate                              as Iterate
import           RL.Matrix                                ( allWithin )

import           RL.Process.Finite                        ( toRewardProcess )

import           RL.Solve.Approximate                     ( evaluateFiniteMRP
                                                          , evaluateMRP
                                                          )

import           Test.Processes                           ( flipFlop )

import           Test.Tasty
import           Test.Tasty.HUnit

tests :: TestTree
tests = testGroup
  "Approximate"
  [ testGroup
    "evaluateFiniteMRP"
    [ testCase "flipFlop" $ do
        let v₀     = Approx.linear [bool 0 1, bool 1 0]
            vs     = evaluateFiniteMRP (flipFlop 0.7) 0.99 v₀
            Just v = Iterate.converge (Approx.within 1e-5) vs
        assertWithin 0.1 (Approx.eval' v [True, False]) (scalar 170)
    ]
  , testGroup
    "evaluateMRP"
    [ testCase "flipFlop" $ do
        let v₀      = Approx.linear [bool 0 1, bool 1 0]
            states  = bernoulli 0.5
            process = toRewardProcess $ flipFlop 0.7
            vs      = evaluateMRP process states 0.99 10 v₀

        vs' <- sampleIO $ Streaming.toList_ $ Streaming.take 100 vs
        print [ Approx.eval' v [True, False] | v <- vs' ]

        Just v <- sampleIO $ Iterate.converge' (Approx.within 1e-3) vs
        assertWithin 0.1 (Approx.eval' v [True, False]) (scalar 170)
    ]
  ]

  -- TODO: Move to shared module somewhere?
assertWithin ϵ v₁ v₂ = assertBool message $ allWithin ϵ v₁ v₂
  where message = printf "%s is not within %f of %s" (show v₁) ϵ (show v₂)
