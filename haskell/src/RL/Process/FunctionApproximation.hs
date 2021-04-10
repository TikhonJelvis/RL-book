{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE NamedFieldPuns #-}
module RL.Process.FunctionApproximation where

import           Data.Default                             ( Default(..) )
import qualified Data.Vector                              ( Vector )
import qualified Data.Vector.Storable                    as Vector

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , (<.>)
                                                          , R
                                                          , Vector
                                                          , scalar
                                                          , scale
                                                          , size
                                                          , tr'
                                                          , (|>)
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import qualified RL.Matrix                               as Matrix

-- * Function Approximation

-- | Approximations for functions of the type a → ℝ.
class Approx f where
  -- | Evaluate the function approximation as a function.
  eval :: f a -> a -> R

  -- | Improve the function approximation given a list of
  -- observations.
  update :: f a -> [(a, R)] -> f a

  -- | Are the two function approximations within ϵ of each other?
  within :: R -> f a -> f a -> Bool


-- * Weighted Approximations

-- ** Learning Rates

data Adam = Adam
  { α      :: !R
  , decay1 :: !R
  , decay2 :: !R
  }
  deriving (Show, Eq)

instance Default Adam where
  def = Adam { α = 0.001, decay1 = 0.9, decay2 = 0.999 }

data AdamCache = AdamCache
  { cache1 :: !(Vector R)
  , cache2 :: !(Vector R)
  }
  deriving (Show, Eq)

adamCache :: Int -> AdamCache
adamCache n = AdamCache { cache1 = Vector.replicate n 0, cache2 = Vector.replicate n 0 }

-- ** Weights

-- | A vector of weights that can be updated, using Adam to manage
-- their learning rate. These can be a layer of a neural net or a
-- standalone linear approximation.
data Weights = Weights
  { values :: !(Vector R)
  , time   :: !Int
  , adam   :: !Adam
  , cache  :: !AdamCache
  }
  deriving (Show, Eq)

-- | Create a new 'Weights' value with 'time' at 0 and reasonable
-- defaults for 'adam' and 'cache'.
weights :: Vector R -> Weights
weights values =
  Weights { values, time = 0, adam = def, cache = adamCache (size values) }

-- | Are the two sets of weights within ϵ of each other?
weightsWithin :: R -> Weights -> Weights -> Bool
weightsWithin ϵ w₁ w₂ = Matrix.allWithin ϵ (values w₁) (values w₂)

updateWeights :: Weights -> Vector R -> Weights
updateWeights = undefined

-- ** Linear Approximations

data Linear a = Linear
  { ϕ :: (a -> Vector R)
    -- ^ Get all the features for an input @a@.
  , w :: Weights
    -- ^ The weights of all the features ϕ. Should have the same
    -- dimension as ϕ returns.
  , λ :: R
    -- ^ The regularization coefficient.
  }

-- | Convert a list of feature functions into a single function
-- returning a vector.
-- 
-- Basic idea: @[f₁, f₂, f₃]@ becomes @λ x → <f₁ x, f₂ x, f₃ x>@
features :: [a -> R] -> (a -> Vector R)
features fs a = length fs |> [ f a | f <- fs ]

lossGradient :: Linear a -> [(a, R)] -> Vector R
lossGradient Linear { ϕ, w, λ } xy = (tr' ϕₓ #> (y' - y)) / n + scale λ (values w)
 where
  (x, Matrix.fromList -> y) = unzip xy
  ϕₓ                        = Matrix.fromRows (ϕ <$> x)
  y'                        = ϕₓ #> values w
  n                         = scalar (fromIntegral $ length xy)

instance Approx Linear where
  eval Linear { ϕ, w } a = ϕ a <.> values w

  update linear@Linear { w } xy = linear { w = updateWeights w $ lossGradient linear xy }

  within ϵ l₁ l₂ = weightsWithin ϵ (w l₁) (w l₂)
