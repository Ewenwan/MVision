/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

MapPoint::MapPoint()
: id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), observed_times_(0)
{

}

MapPoint::MapPoint ( long id, Vector3d position, Vector3d norm )
: id_(id), pos_(position), norm_(norm), observed_times_(0)
{

}

MapPoint::Ptr MapPoint::createMapPoint()
{
    static long factory_id = 0;
    return MapPoint::Ptr( 
        new MapPoint( factory_id++, Vector3d(0,0,0), Vector3d(0,0,0) )
    );
}

}
