# OpenGV 视觉几何问题开源库  geometric vision problems
[代码](https://github.com/Ewenwan/opengv)


目录：

## 计算相机绝对位姿 Absolute camera pose computation:
        1. absolute pose computation with known rotation
        2. two P3P-algorithms (Kneip, Gao)
        3. generalized P3P
        4. the EPnP algorithm by Lepetit and Fua
        5. an extension of the EPnP algorithm to the non-central case (Kneip)
        6. the generalized absolute pose solver presented at ICRA 2013 (Kneip)
        7. non-linear optimization over n correspondences (both central and non-central)
        8. the UPnP algorithm presented at ECCV 2014 (both central and non-central, and minimal and non-minimal)
        
## 计算相机相对位姿 Relative camera-pose computation:
        1. 2-point algorithm for computing the translation with known relative rotation
        2. 2-point algorithm for deriving the rotation in a pure-rotation situation
        3. n-point algorithm for deriving the rotation in a pure-rotation situation
        4. 5-point algorithm by Stewenius
        5. 5-point algorithm by Nister
        6. 5-point algorithm to solve for rotations directly (by Kneip)
        7. 7-point algorithm
        8. 8-point algorithm by Longuet-Higgins
        9. 6-point algorithm by Henrik Stewenius for generalized relative pose
        10. 17-point algorithm by Hongdong Li
        11. non-linear optimization over n correspondences (both central and non-central)
        11. relative rotation as an iterative eigenproblem (by Kneip)
        12. generalized reltive rotation for multi-camera systems as an iterative eigenproblem (by Kneip)
## Two methods for point-triangulation      三角化
## Arun's method for aligning point clouds  点云配准
## Generic sample-consensus problems for most algorithms useable with Ransac 随机采样序列
## Math tools:
        Generic Sturm-sequence implementation for numerical root-finding
        Algebraic root finding
        Cayley rotations
    Unit/Benchmarking tests for all algorithms
    Matlab interface
    Python interface
