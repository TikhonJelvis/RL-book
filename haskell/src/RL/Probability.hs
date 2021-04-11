module RL.Probability where

import           Control.Monad.Bayes.Enumerator           ( Enumerator )
import qualified Control.Monad.Bayes.Enumerator          as Enumerator

import           Numeric.LinearAlgebra                    ( R )

-- | Probability monads where we can (approximate) the expected value
-- of distributions.
class MonadExpect m where
  expected :: (a -> R) -> m a -> m R

instance MonadExpect Enumerator where
  expected f = pure . Enumerator.expectation (log . f)
