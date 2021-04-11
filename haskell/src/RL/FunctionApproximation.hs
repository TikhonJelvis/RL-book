{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE NamedFieldPuns #-}
module RL.FunctionApproximation where

import           Data.Default                             ( Default(..) )
import qualified Data.Vector                             as V
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

import           RL.Matrix                                ( (<$$>) )
import qualified RL.Matrix                               as Matrix

-- * Function Approximation

-- | Approximations for functions of the type a → ℝ.
class Approx f where
  -- | Evaluate the function approximation as a function.
  eval :: f a -> a -> R

  -- | Improve the function approximation given a list of
  -- observations.
  update :: f a -> V.Vector a -> Vector R -> f a

  -- | Are the two function approximations within ϵ of each other?
  within :: R -> f a -> f a -> Bool


-- | Evaluate a whole bunch of inputs and produce a vector of the
-- results.
eval' :: Approx f => f a -> V.Vector a -> Vector R
eval' f xs = Matrix.storable $ eval f <$> xs

-- * Weighted Approximations

-- ** Learning Rates

data Adam = Adam
  { α  :: !R
  , β₁ :: !R
  , β₂ :: !R
  }
  deriving (Show, Eq)

instance Default Adam where
  def = Adam { α = 0.001, β₁ = 0.9, β₂ = 0.999 }

data AdamCache = AdamCache
  { cache1 :: !(Vector R)
  , cache2 :: !(Vector R)
  }
  deriving (Show, Eq)

-- | Create an 'AdamCache' for a set of @n@ weights, initialized to 0.
adamCache :: Int -> AdamCache
adamCache n = AdamCache { cache1 = Vector.replicate n 0, cache2 = Vector.replicate n 0 }

-- | Update both Adam caches given 'Adam' settings and a loss
-- gradient.
updateCache :: AdamCache
            -- ^ The Adam cache to update.
            -> Adam
            -- ^ Adam settings to use.
            -> Vector R
            -- ^ A vector representing the loss gradient (see
            -- 'lossGradient').
            -> AdamCache
updateCache AdamCache { cache1, cache2 } Adam { α, β₁, β₂ } grad = AdamCache
  { cache1 = scale β₁ cache1 + scale (1 - β₁) grad
  , cache2 = scale β₂ cache2 + scale (1 - β₂) (grad ** 2)
  }

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

-- | Update weights given a vector representing the loss gradient.
updateWeights :: Weights -> Vector R -> Weights
updateWeights weights grad = weights { time   = time weights + 1
                                     , cache  = cache'
                                     , values = values'
                                     }
 where
  values'            = values weights - scale α m' / (sqrt v' + ϵ)
  m'                 = cache1 cache' / scalar (1 - β₁ ** t)
  v'                 = cache2 cache' / scalar (1 - β₂ ** t)
  t                  = fromIntegral (time weights + 1)
  ϵ                  = 1e-6

  cache'             = updateCache (cache weights) (adam weights) grad
  Adam { α, β₁, β₂ } = adam weights

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

lossGradient :: Linear a -> V.Vector a -> Vector R -> Vector R
lossGradient Linear { ϕ, w, λ } x y = (tr' ϕₓ #> (y' - y)) / n + scale λ (values w)
 where
  ϕₓ = ϕ <$$> x
  y' = ϕₓ #> values w
  n  = scalar (fromIntegral $ length x)

instance Approx Linear where
  eval Linear { ϕ, w } a = ϕ a <.> values w

  update linear@Linear { w } x y =
    linear { w = updateWeights w $ lossGradient linear x y }

  within ϵ l₁ l₂ = weightsWithin ϵ (w l₁) (w l₂)
