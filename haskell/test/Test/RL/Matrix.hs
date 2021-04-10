module Test.RL.Matrix where

import           Control.Monad                            ( replicateM )

import           Numeric.LinearAlgebra                    ( Matrix
                                                          , R
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Matrix                                ( allWithin
                                                          , matrixRows
                                                          , sumRows
                                                          )

import           Test.Tasty
import           Test.Tasty.HUnit
import           Test.Tasty.QuickCheck

tests :: TestTree
tests = testGroup "Matrix" [prop_sumRows, test_allWithin]

prop_sumRows :: TestTree
prop_sumRows = testProperty "sumRows" $ forAll matrixRows $ \rows ->
  sumRows (Matrix.fromLists rows) == Matrix.fromList (sum <$> rows)

test_allWithin :: TestTree
test_allWithin = testGroup
  "allWithin"
  [ testProperty "∀a. allWithin ϵ a a" $ \(NonNegative ϵ) ->
    forAll matrixRows $ \rows -> let m = Matrix.fromLists rows in allWithin ϵ m m
  , testProperty "∀a. a ≠ b ⇒ ¬allWithin 0 a b" $ \(Positive n) (Positive m) ->
    forAll (mat m n) $ \a -> forAll (mat m n) $ \b -> a /= b ==> not (allWithin 0 a b)
  , testProperty "∀a. allWithin ϵ a (a + ϵ)" $ \(Positive ϵ) ->
    forAll matrixRows $ \rows ->
      let a = Matrix.fromLists rows in allWithin (1.001 * ϵ) a (a + Matrix.scalar ϵ)
  ]
 where
  mat :: Int -> Int -> Gen (Matrix R)
  mat m n = Matrix.fromLists <$> replicateM m (replicateM n arbitrary)
