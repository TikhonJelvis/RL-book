module RL.Probability where

import           Control.Monad                            ( replicateM )
import           Control.Monad.Bayes.Class                ( MonadSample )

import           Numeric.LinearAlgebra                    ( R )

-- | Approximate the expected value of @f(X)@ for some random variable
-- @X@ by drawing @n@ samples.
expected :: MonadSample m => Int -> (a -> R) -> m a -> m R
expected n f x = do
  samples <- replicateM n (f <$> x)
  pure (sum samples / fromIntegral n)
