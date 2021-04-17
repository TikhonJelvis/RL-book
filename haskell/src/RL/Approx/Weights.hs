{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Vectors of /weights/ as used in linear and neural approximations.
module RL.Approx.Weights where

import           Data.Default                             ( Default(..) )
import qualified Data.Vector.Storable                    as Vector

import           Numeric.LinearAlgebra                    ( R
                                                          , Vector
                                                          , scalar
                                                          , scale
                                                          , size
                                                          )

import qualified RL.Matrix                               as Matrix
import           RL.Vector                                ( Affine(..) )

-- * Learning Rates

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
adamCache n =
  AdamCache { cache1 = Vector.replicate n 0, cache2 = Vector.replicate n 0 }

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

-- * Weights

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

instance Affine Weights where
  type Diff Weights = Vector R

  w₁ .-. w₂ = values w₁ - values w₂

  w .+ v = w { values = values w + v }

-- | Create a new 'Weights' value with 'time' at 0 and reasonable
-- defaults for 'adam' and 'cache'.
create :: Vector R -> Weights
create values =
  Weights { values, time = 0, adam = def, cache = adamCache (size values) }

-- | Initialize a set of weights with the given dimension and starting
-- value.
init :: Int
     -- ^ Dimension (number of weights)
     -> R
     -- ^ Starting value
     -> Weights
init n z = create $ Vector.replicate n z

-- | Are the two sets of weights (@v@ and @w@) within @ϵ@ of each
-- other?
--
-- @
-- ∀i. |vᵢ - wᵢ| ≤ ϵ
-- @
within :: R
              -- ^ ϵ
       -> Weights
              -- ^ v
       -> Weights
              -- ^ w
       -> Bool
within ϵ w₁ w₂ = Matrix.allWithin ϵ (values w₁) (values w₂)

-- | Update weights given a vector representing the loss gradient.
update :: Weights
              -- ^ The weights value to update.
       -> Vector R
              -- ^ The gradient to update along. Should have the same
              -- dimension as @w@ for the 'Weights' being updated.
       -> Weights
update weights grad = weights { time   = time weights + 1
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

