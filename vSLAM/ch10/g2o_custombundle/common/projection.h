#ifndef PROJECTION_H
#define PROJECTION_H

#include "tools/rotation.h"

// camera : 9 dims array with 
// [0-2] : angle-axis rotation 
// [3-5] : translateion
// [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
// point : 3D location. 
// predictions : 2D predictions with center of the image plane. 

template<typename T>
inline bool CamProjectionWithDistortion(const T* camera, const T* point, T* predictions){
    // Rodrigues' formula
    T p[3];
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center fo distortion
    T xp = -p[0]/p[2];
    T yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion
    const T& l1 = camera[7];
    const T& l2 = camera[8];

    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    const T& focal = camera[6];
    predictions[0] = focal * distortion * xp;
    predictions[1] = focal * distortion * yp;

    return true;
}



#endif // projection.h