{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
-- | Vector spaces as an algebraic structure.
--
-- Heavily inspired by classes in the @linear@ package but without
-- needing an underlying 'Functor' instance, letting us provide
-- instances for @hmatrix@ types, function approximations... etc.
module RL.Vector where

import           Numeric.LinearAlgebra                    ( Container
                                                          , R
                                                          , Vector
                                                          , scalar
                                                          , scale
                                                          )

-- * Additive Groups

-- | An additive group.
class Additive v where
  zero :: v
  (+:) :: v -> v -> v

  (-:) :: v -> v -> v
  v₁ -: v₂ = v₁ +: (inv v₂)

  inv :: v -> v
  inv v = zero -: v

  {-# MINIMAL zero, (+:), ((-:) | inv) #-}

instance Additive R where
  zero = 0
  (+:) = (+)
  (-:) = (-)
  inv  = negate

instance Additive (Vector R) where
  zero = scalar 0
  (+:) = (+)
  (-:) = (-)
  inv  = negate


-- * Vector Spaces

-- | A vector space is an additive group that can be multiplied by
-- scalars from some field (denoted by the type @Scalar v@ here).
class Additive v => VectorSpace v where
  -- | The type of scalars for this vector space. Should be a field.
  type Scalar v

  -- | Scalar multiplication. Should be distributive with @Scalar v@'s
  -- field multiplication.
  (*:) :: Scalar v -> v -> v

instance VectorSpace R where
  type Scalar R = R
  (*:) = (*)

instance VectorSpace (Vector R) where
  type Scalar (Vector R) = R
  (*:) = scale


-- * Affine Space

-- | Affine spaces are, conceptually, vector spaces "without" an
-- origin—or, more specifically, vector spaces where the origin can be
-- any arbitrary point.
--
-- Affine spaces have an associated vector space (@Diff p@) that
-- corresponds to translations between elements of @p@.
--
-- An example is points on a plane. Choosing a point on a plane to act
-- as an origin is arbitrary, so we should think of points as vectors
-- /without/ a specific 0 element. However, there is an associated
-- vector space of "differences" (ie /translations/) between
-- points. This space /does/ have a natural 0 element—the translation
-- (0, 0) between a point and itself.
class Additive (Diff p) => Affine p where
  -- | The type of /differences/ between points in an affine
  -- space. Should form a vector space.
  type Diff p

  -- | The difference between two points in the affine space.
  (.-.) :: p -> p -> Diff p

  -- | Add a point and a vector.
  (.+) :: p -> Diff p -> p
  p .+ d = d +. p

  -- | Add a vector and a point.
  (+.) :: Diff p -> p -> p
  d +. p = p .+ d

  {-# MINIMAL (.-.), ((+.) | (.+)) #-}

-- | Subtract a vector from a point.
(.-) :: Affine p => p -> Diff p -> p
p .- d = p .+ inv d

instance Affine R where
  type Diff R = R
  (.-.) = (-)
  (.+)  = (+)

-- | A type for points that have the same underlying representation as
-- some kind of vector. For example, points on a plane look the same
-- as vectors in ℝ² but don't have the same semantics.
newtype Point t = Point { toVector :: t }
  deriving stock (Show, Eq)

instance Additive t => Affine (Point t) where
  type Diff (Point t) = t

  Point p₁ .-. Point p₂ = p₁ -: p₂

  Point p .+ d = Point (p +: d)
