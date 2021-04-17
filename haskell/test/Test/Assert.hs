module Test.Assert where

import           Text.Printf                              ( printf )

import           RL.Within                                ( Within(..) )

import           Test.Tasty
import           Test.Tasty.HUnit

assertWithin :: (Show a, Within a) => Double -> a -> a -> Assertion
assertWithin ϵ x₁ x₂ = assertBool message $ within ϵ x₁ x₂
 where
  message = printf "%s is not within %s of %s" (show x₁) (show ϵ) (show x₂)
