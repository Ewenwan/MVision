# OpenGV 视觉几何问题开源库  geometric vision problems

目录：

## Absolute camera pose computation:
        absolute pose computation with known rotation
        two P3P-algorithms (Kneip, Gao)
        generalized P3P
        the EPnP algorithm by Lepetit and Fua
        an extension of the EPnP algorithm to the non-central case (Kneip)
        the generalized absolute pose solver presented at ICRA 2013 (Kneip)
        non-linear optimization over n correspondences (both central and non-central)
        the UPnP algorithm presented at ECCV 2014 (both central and non-central, and minimal and non-minimal)
## Relative camera-pose computation:
        2-point algorithm for computing the translation with known relative rotation
        2-point algorithm for deriving the rotation in a pure-rotation situation
        n-point algorithm for deriving the rotation in a pure-rotation situation
        5-point algorithm by Stewenius
        5-point algorithm by Nister
        5-point algorithm to solve for rotations directly (by Kneip)
        7-point algorithm
        8-point algorithm by Longuet-Higgins
        6-point algorithm by Henrik Stewenius for generalized relative pose
        17-point algorithm by Hongdong Li
        non-linear optimization over n correspondences (both central and non-central)
        relative rotation as an iterative eigenproblem (by Kneip)
        generalized reltive rotation for multi-camera systems as an iterative eigenproblem (by Kneip)
    Two methods for point-triangulation
    Arun's method for aligning point clouds
    Generic sample-consensus problems for most algorithms useable with Ransac
    Math tools:
        Generic Sturm-sequence implementation for numerical root-finding
        Algebraic root finding
        Cayley rotations
    Unit/Benchmarking tests for all algorithms
    Matlab interface
    Python interface
