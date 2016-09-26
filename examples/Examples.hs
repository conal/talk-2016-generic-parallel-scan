-- -*- flycheck-disabled-checkers: '(haskell-ghc haskell-stack-ghc); -*-
{-# LANGUAGE PatternSynonyms, ConstraintKinds, ScopedTypeVariables, AllowAmbiguousTypes #-}
{-# LANGUAGE CPP, LambdaCase, GADTs, TypeOperators, DataKinds, TypeApplications, TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Example of Haskell to hardware via reification and CCC

{-# OPTIONS_GHC -Wall -fno-warn-unticked-promoted-constructors -fno-warn-missing-signatures #-}

-- {-# OPTIONS -fplugin-opt=ReificationRules.Plugin:trace #-}

{-# OPTIONS_GHC -fno-warn-unused-imports #-} -- TEMP
{-# OPTIONS_GHC -fno-warn-unused-binds   #-} -- TEMP

module Main where

import Prelude hiding (zipWith)

import Control.Arrow ((***))
import Data.Tuple (swap)
import GHC.Generics hiding (S,C)

import Data.Key
import Data.Pointed

import Circat.Complex

import ReificationRules.Run (go,goSep)
import ReificationRules.Misc ((:*),Unop,transpose,unknown)

import ShapedTypes.Nat
import ShapedTypes.Pair
import ShapedTypes.Vec
import ShapedTypes.LPow (LPow)
import ShapedTypes.RPow (RPow)
import qualified ShapedTypes.LPow as L
import qualified ShapedTypes.RPow as R
import qualified ShapedTypes.Fams as F
import ShapedTypes.Fams (LVec,RVec)
import ShapedTypes.ScanF
import ShapedTypes.Orphans ()

-- GADT versions
type RBin = RPow Pair
type LBin = LPow Pair

-- Type family versions
type RBin' n = F.RPow Pair n
type LBin' n = F.LPow Pair n

main :: IO ()
main = do
  return ()

--   -- hangs
--   goSep "foo" 1 (powers @(RBin' N0) @Int)

--   -- hangs
--   goSep "foo" 1 (point @(RBin' Z) @Int)

--   -- fine
--   goSep "foo" 1 (fmap @(RBin' Z) not)

--   -- fine
--   goSep "foo" 1 (id :: RBin' Z Int -> Par1 Int)

--   -- fine
--   goSep "foo" 1 (point @Par1 @Int)

--   -- fine
--   goSep "foo" 1 (lproducts @(RBin' N0) @Int)

--   goSep "foo" 1 (point @(RBin' N1))

--   -- fine
--   goSep "foo" 1.0 (lsums @(RBin' N4) @Int)

--   -- fine
--   goSep "foo" 1.0 (lsums @(F.LVec N5) @Int)

--------

--   goSep "mult-table-rb3-rb3" 10 (multTable @(RBin N3) @(RBin N3) @Int)

--   goSep "point-rb4" 2 (point @(RBin N4) @Int)
--   goSep "dot-rb4-1" 2 ((<.>) @(And1 (RBin N4)) @Int)
--   goSep "lproducts-rb4" 2 (lproducts @(RBin N4) @Int)
--   goSep "powers-rb4" 6 (powers @(RBin N4) @Int)
--   goSep "powers-decoupled-rb4" 2 (lproducts . (unknown :: Unop (RBin N4 Int)) . point)
--   goSep "evalPoly-rb4" 3 (evalPoly @(RBin N4) @Int)

--   goSep "point-lb4" 2 (point @(LBin N4) @Int)
--   goSep "dot-lb4-1" 2 ((<.>) @(And1 (LBin N4)) @Int)

--   goSep "lproducts-lb4" 2 (lproducts @(LBin N4) @Int)
--   goSep "powers-lb4" 6 (powers @(LBin N4) @Int)
--   goSep "powers-decoupled-lb4" 2 (lproducts . (unknown :: Unop (LBin N4 Int)) . point)
--   goSep "evalPoly-lb4" 3 (evalPoly @(LBin N4) @Int)

--   goSep "lsums-lv0" 1.0 (lsums @(LVec N0) @Int)

-- --   -- For revised parallel scan talk (2016-10-05)
--   goSep "lsums-p"   0.5 (lsums @Pair @Int)
  goSep "lsums-lv4" 1.0 (lsums @(LVec N4) @Int)
--   goSep "lsums-rv4" 2.0 (lsums @(RVec N4) @Int) -- yipes!
--   goSep "lsums-lv16" 0.5 (lsums @(LVec N16) @Int)
--   goSep "lsums-lv5xlv11" 1 (lsums @(LVec N5 :*: LVec N11) @Int)
--   goSep "lsums-lv11xlv5" 0.75 (lsums @(LVec N11 :*: LVec N5) @Int)

--   goSep "lsumsp-u" 0.75 (lsums' @U1 @Int)
--   goSep "lsumsp-p" 0.75 (lsums' @Par1 @Int)

--   goSep "lsums-u"   0.75 (lsums @U1 @Int)
--   goSep "lsums-i"   0.75 (lsums @Par1 @Int)
--   goSep "lsums-uxi" 0.75 (lsums @(U1 :*: Par1) @Int)
--   goSep "lsums-1-0" 0.75 (lsums @(Par1 :*: U1) @Int)
--   goSep "lsums-1-1-0" 0.75 (lsums @(Par1 :*: (Par1 :*: U1)) @Int)

--   goSep "lsums-lpow-4-2" 1 (lsums @(LPow (LVec N4) N2) @Int)
--   goSep "lsums-rpow-4-2" 1 (lsums @(RPow (LVec N4) N2) @Int)

--   goSep "unknown-lv4-lv6" 1 (unknown :: LVec N4 Int -> LVec N6 Bool)

--   goSep "lsums-0-1-1-1-1-l" 1 (lsums @((((U1 :*: Par1) :*: Par1) :*: Par1) :*: Par1) @Int)
--   goSep "lsums-1-1-1-1-0-r" 1 (lsums @(Par1 :*: (Par1 :*: (Par1 :*: (Par1 :*: U1)))) @Int)

--   goSep "lsums-lv5-5-6-l" 1 (lsums @((LVec N5 :*: LVec N5) :*: LVec N6) @Int)
--   goSep "lsums-lv5-5-6-r" 1 (lsums @(LVec N5 :*: (LVec N5 :*: LVec N6)) @Int)

--   goSep "lsums-lv1-7" 0.5 (lsums @(LVec N1 :*: LVec N7) @Int)
--   goSep "lsums-lv1-15" 0.75 (lsums @(LVec N1 :*: LVec N15) @Int)

--   goSep "lsumsp-lv8-and-lv8" 1 (lsums' @(LVec N8) @Int *** lsums' @(LVec N8) @Int)

--   goSep "lsumsp-lv8-lv8-unknown" 1
--     (unknown . (lsums' *** lsums') :: LVec N8 Int :* LVec N8 Int -> LVec N16 Int)

--   goSep "lsums-lv5"  0.5 (lsums @(LVec N5 ) @Int)
--   goSep "lsums-lv11" 0.5 (lsums @(LVec N11) @Int)

--   goSep "lsums-rv16" 1 (lsums @(RVec N16) @Int)

--   goSep "lsums-lv8-lv8-unknown" 1
--     (unknown . (lsums *** lsums) :: LVec N8 Int :* LVec N8 Int -> And1 (LVec N16) Int)

--   goSep "lsums-lv3olv4" 1.2 (lsums @(LVec N3 :.: LVec N4) @Int)

--   goSep "lsums-lv5olv7" 1.2 (lsums @(LVec N5 :.: LVec N7) @Int)

--   goSep "lsums-lv3olv4-unknown" 1
--     (unknown . fmap lsums :: LVec N3 (LVec N4 Int) -> And1 (LVec N3 :.: LVec N4) Int)

--   goSep "lsums-lv4olv4-unknown" 1
--     (unknown . fmap lsums :: LVec N4 (LVec N4 Int) -> And1 (LVec N4 :.: LVec N4) Int)

--   goSep "lsums-lv5-lv11-unknown" 1.0
--     (unknown . (lsums *** lsums) :: LVec N5 Int :* LVec N11 Int -> And1 (LVec N16) Int)

--   goSep "lsumsp-lv4-5-7-l" 1 (lsums' @((LVec N4 :*: LVec N5) :*: LVec N7) @Int)
--   goSep "lsumsp-lv4-5-7-r" 1 (lsums' @(LVec N4 :*: (LVec N5 :*: LVec N7)) @Int)

--   goSep "lsumsp-lv5-5-6-l" 1 (lsums' @((LVec N5 :*: LVec N5) :*: LVec N6) @Int)
--   goSep "lsumsp-lv5-5-6-r" 1 (lsums' @(LVec N5 :*: (LVec N5 :*: LVec N6)) @Int)

--   goSep "lsumsp-lv5xlv11" 1 (lsums' @(LVec N5 :*: LVec N11) @Int)

  -- The next are logically identical, but the commutation optimization in
  -- Circat can make them appear to differ slightly. For comparison, I turn on
  -- NoCommute in Circat.hs.
--   goSep "lsumsp-lv8xlv8" 1 (lsums' @(LVec N8 :*: LVec N8) @Int)
--   goSep "lsumsp-p-lv8" 1 (lsums' @(Pair :.: LVec N8) @Int)
--   goSep "lsumsp-lv8-p" 1 (lsums' @(LVec N8 :.: Pair) @Int)
--   goSep "lsumsp-lv16" 0.5 (lsums' @(LVec N16) @Int)
--   goSep "lsumsp-lv4olv4" 1 (lsums' @(LVec N4 :.: LVec N4) @Int)
--   goSep "lsumsp-lb4" 1 (lsums' @(LBin N4) @Int)
--   goSep "lsumsp-rb4" 1 (lsums' @(RBin N4) @Int)
--   goSep "lsumsp-bush2" 1 (lsums' @(F.Bush N2) @Int)

--   goSep "lsums-rb3" 1.5 (lsums @(RBin N3) @Int)

--   -- 0:0.5; 1:0.75; 2:1.5; 3:8
--   goSep "sum-bush0" 0.5  (sum @(F.Bush N0) @Int)
--   goSep "sum-bush1" 0.75 (sum @(F.Bush N1) @Int)
--   goSep "sum-bush2" 1.5  (sum @(F.Bush N2) @Int)
--   goSep "sum-bush3" 8    (sum @(F.Bush N3) @Int)

--   goSep "power-p"   1 (power @Pair      @Int)
--   goSep "power-rb4" 1 (power @(RBin N4) @Int)
--   goSep "power-lb4" 1 (power @(LBin N4) @Int)

--   -- 0:0.75; 1:1; 2:2; 3:16
--   goSep "lsums-bush0" 0.75 (lsums @(F.Bush N0) @Int)
--   goSep "lsums-bush1" 1    (lsums @(F.Bush N1) @Int)
--   goSep "lsums-bush2" 2    (lsums @(F.Bush N2) @Int)
--   goSep "lsums-bush3" 8    (lsums @(F.Bush N3) @Int)

   -- 1:0.5; 2:0.75; 3:1; 4:4, 5:8
--    goSep "lsums-bushp1" 0.5  (lsums @(F.Bush N1) @Int)
--    goSep "lsums-bushp2" 0.75 (lsums @(F.Bush N2) @Int)
--    goSep "lsums-bushp3" 1    (lsums @(F.Bush N3) @Int)
--    goSep "lsums-bushp4" 4    (lsums @(F.Bush N4) @Int)
--    goSep "lsums-bushp5" 16   (lsums @(F.Bush N5) @Int)

{--------------------------------------------------------------------
    Example helpers
--------------------------------------------------------------------}

evalPoly :: (LScan f, Foldable f, Zip f, Pointed f, Num a)
         => And1 f a -> a -> a
evalPoly coeffs x = coeffs <.> powers x

(<.>) :: (Foldable f, Zip f, Num a) => f a -> f a -> a        -- dot product
u <.> v = sum (zipWith (*) u v)

multTable :: forall g f a. (LScan g, Pointed g, LScan f, Pointed f, Num a)
                    => And1 g (And1 f a)
multTable  = multiples  <$> multiples  1

