#include <pangolin/pangolin.h>
#include <float.h>
#include <iostream>
#include <cmath>
#define  FORKLIFT_LENGTH  1.3;
#define  D  (1.5)
#define  WHEEL_WIDTH 0.3
#define  WHEEL_LEN 0.2
void DrawRearSteerWheel(float steerAngle)
{

    pangolin::OpenGlMatrix Twc,Twc_T ,Twc_R;
    Twc.SetIdentity();

    Twc_R =Twc.RotateZ(steerAngle);

    Twc_R(0,3) += Twc(0,3);
    Twc_R(1,3) += Twc(1,3);
    Twc_R(2,3) += Twc(2,3);

    glPushMatrix();
    glMultMatrixd(Twc_R.m);


      glBegin(GL_LINES);
      glLineWidth(1);
      glColor3f(1, 0, 0);

      glVertex3f(-WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(WHEEL_LEN ,WHEEL_WIDTH, 0);

      glVertex3f(-WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(-WHEEL_LEN, -WHEEL_WIDTH, 0);

      glVertex3f(WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(WHEEL_LEN, -WHEEL_WIDTH, 0);

      glVertex3f(WHEEL_LEN,-WHEEL_WIDTH, 0);
      glVertex3f(-WHEEL_LEN, -WHEEL_WIDTH, 0);

      glEnd();

  glPopMatrix();

}


void DrawPassiveWheel()
{
   float baseY  = FORKLIFT_LENGTH ;
   float baseLX =  -D/2;
   float baseLR=    D/2;


   glPointSize(5);
   glBegin(GL_POINTS);

    glColor3f(1, 0, 0);
    // DRAW LEFT

   glVertex3f( -0.5,1.3, 0);

   glVertex3f(0.5, 1.3, 0);

   glEnd();

   glBegin(GL_LINES);

   glVertex3f(-0.7, 1.5, 0);
   glVertex3f(0.7, 1.5, 0);

   glVertex3f(-0.7, -0.4, 0);
   glVertex3f(0.7, -0.4, 0);

   glVertex3f(-0.7, 1.5, 0);
    glVertex3f(-0.7, -0.4, 0);

    glVertex3f(0.7, 1.5, 0);
    glVertex3f(0.7, -0.4, 0);
   glEnd();

}

void DrawForkLift(float x, float y, float yaw,float steer)
{
    pangolin::OpenGlMatrix Twc,Twc_T ,Twc_R;
    Twc.SetIdentity();
    Twc_T = Twc.Translate(x, y, 0.0);
    Twc_R =Twc.RotateZ(yaw);

    Twc_R(0,3) += Twc_T(0,3);
    Twc_R(1,3) += Twc_T(1,3);
    Twc_R(2,3) += Twc_T(2,3);

    glPushMatrix();
    glMultMatrixd(Twc_R.m);
    DrawPassiveWheel();
    DrawRearSteerWheel(steer);
    glPopMatrix();
}
void drawGrid(int size)
{
    glBegin(GL_LINES);
    glLineWidth(1);

    glColor3f(0.1, 0.1, 0.1); //gray

    for(int i = -size; i <= size ; i++){

      glVertex3f(i,size,  0);
      glVertex3f(i, -size, 0);
      glVertex3f( size, i, 0);
      glVertex3f(-size, i, 0);
    }
    glEnd();
}
using namespace std;
float k = 0.7; // look forward gain
float Lfc = 2.0; // look ahead distance
float Kp = 1.0; // speed propotional gain
float dt = 0.1;
float L = 1.3;
class Point2f
{
public:
    Point2f()
    {x=0; y= 0;}
    Point2f(float _x, float _y )
    {
       x = _x;
       y = _y;
    }
    float x;

    float y;
};

class State
{
public:
    State()
    {
        x= 0;
        y = 0;
        yaw = 0;
        v = 0;
    }
    void update(float a, float delta)
    {
        x   = x   + v*cos(yaw)*dt;
        y   = y   + v*sin(yaw)*dt;
        yaw = yaw + v/L*tan(delta)*dt;
        v   = v   + a*dt;
    }
    float x;
    float y;
    float yaw;
    float v;
};

float PIDcontrol(float target, float current)
{
    float a = Kp*(target - current);
    return a;
}

int calc_target_index(State state, vector<Point2f>path)
{
    float minDis = FLT_MAX;
    int ind = 0;
    for(int i=0; i<path.size(); i++)
    {
        float dx = state.x - path[i].x;
        float dy = state.y - path[i].y;
        float dis = sqrt(dx*dx + dy*dy);
        if(dis<minDis)
        {
            minDis = dis;
            ind = i;
        }
    }
    float Ld = 0.0;
    float Lf = k*state.v + Lfc;
    while((Lf>Ld) && (ind+1)<path.size()-1)
    {
        float dx = path[ind+1].x - path[ind].x;
        float dy = path[ind+1].y - path[ind].y;
        Ld += sqrt(dx*dx + dy*dy);
        ind += 1;
    }
    return ind;
}


float pure_pursuit_control(State state, vector<Point2f>path, int pind, float &delta)
{
    int ind = calc_target_index(state, path);
    float tx, ty;
    int N = path.size();
    if(pind >= ind)
    {
        ind = pind;
    }
    if(ind<N)
    {
        tx = path[ind].x;
        ty = path[ind].y;
    }
    else
    {
        tx = path[N-1].x;
        ty = path[N-1].y;
        ind = N-1;
    }
   float alpha =  atan2(ty-state.y, tx-state.x) - state.yaw;
   if(state.v<0)
       alpha = 3.14 - alpha;

   float Lf = k*state.v + Lfc;
   delta = atan2(2.0*L*sin(alpha)/Lf, 1.0);
   return ind;

}
void drawPath(vector<Point2f>path)
{
    glPointSize(4);
    glBegin(GL_POINTS);
    glColor3f(0, 0, 0);
    for(int i=0; i<path.size(); i++)
    {
        glVertex2f(path[i].x, path[i].y);
    }
    glEnd();
}
void drawTarget( Point2f target)
{

    glPointSize(5);
    glBegin(GL_POINTS);
    glColor3f(0, 0, 1);
    glVertex2f(target.x, target.y);
    glEnd();
}
int main()
{
    pangolin::CreateWindowAndBind("purepusuit", 1024, 768);
    glEnable(GL_DEPTH_TEST);


    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 368, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, 0.0,20, 0, 0, 0, 0.0, 1.0, 0.0));

    pangolin::View& d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                                                     .SetHandler(new pangolin::Handler3D(s_cam));
    //1, get Path in memory
    vector<Point2f>path;
    for(float i=-3.14; i<=3.14; i+=0.1)
    {
        Point2f pt;
        pt.x = 10*cos(i);
        pt.y = 10*sin(i);
        path.push_back(pt);
    }

//   for(float i=-10; i<20; i+=0.01)
//    {
//        Point2f pt;
//        pt.x = i;
//       pt.y = 10/(1+exp(-i));
//        //pt.y = sin(i+3.14/6)*i/10.0;
//        path.push_back(pt);
//    }
    float target_speed = 0.5;

    State state;
    state.x= -20.0;
    state.y =-3;
    state.yaw = 0;
    int target_ind = calc_target_index(state,path);
    float di = 3.14/2;
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawTarget( path[target_ind]);
        DrawForkLift(state.x, state.y, 3.14/2+state.yaw, di);
        drawGrid(100);
        drawPath(path);
        if(target_ind<path.size()-1){
        float ai = PIDcontrol(target_speed, state.v);

        target_ind = pure_pursuit_control(state, path, target_ind, di);

        state.update(ai, di);

        }
        pangolin::FinishFrame();
    }


    return 0;
}














#if 0
#define  FORKLIFT_LENGTH  1.3;
#define  D  (1.5)
#define  WHEEL_WIDTH 0.3
#define  WHEEL_LEN 0.2

void drawGrid(int size)
{
    glBegin(GL_LINES);
    glLineWidth(1);

    glColor3f(0.1, 0.1, 0.1); //gray

    for(int i = -size; i <= size ; i++){

      glVertex3f(i,size,  0);
      glVertex3f(i, -size, 0);
      glVertex3f( size, i, 0);
      glVertex3f(-size, i, 0);
    }
    glEnd();
}

void DrawRearSteerWheel(float steerAngle)
{

    pangolin::OpenGlMatrix Twc,Twc_T ,Twc_R;
    Twc.SetIdentity();

    Twc_R =Twc.RotateZ(steerAngle);

    Twc_R(0,3) += Twc(0,3);
    Twc_R(1,3) += Twc(1,3);
    Twc_R(2,3) += Twc(2,3);

    glPushMatrix();
    glMultMatrixd(Twc_R.m);


      glBegin(GL_LINES);
      glLineWidth(1);
      glColor3f(1, 0, 0);

      glVertex3f(-WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(WHEEL_LEN ,WHEEL_WIDTH, 0);

      glVertex3f(-WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(-WHEEL_LEN, -WHEEL_WIDTH, 0);

      glVertex3f(WHEEL_LEN, WHEEL_WIDTH, 0);
      glVertex3f(WHEEL_LEN, -WHEEL_WIDTH, 0);

      glVertex3f(WHEEL_LEN,-WHEEL_WIDTH, 0);
      glVertex3f(-WHEEL_LEN, -WHEEL_WIDTH, 0);

      glEnd();

  glPopMatrix();

}


void DrawPassiveWheel()
{
   float baseY  = FORKLIFT_LENGTH ;
   float baseLX =  -D/2;
   float baseLR=    D/2;


   glPointSize(1);
   glBegin(GL_POINTS);

    glColor3f(1, 0, 0);
    // DRAW LEFT

   glVertex3f( -0.5,1.3, 0);

   glVertex3f(0.5, 1.3, 0);

   glEnd();

   glBegin(GL_LINES);

   glVertex3f(-0.7, 1.5, 0);
   glVertex3f(0.7, 1.5, 0);

   glVertex3f(-0.7, -0.4, 0);
   glVertex3f(0.7, -0.4, 0);

   glVertex3f(-0.7, 1.5, 0);
    glVertex3f(-0.7, -0.4, 0);

    glVertex3f(0.7, 1.5, 0);
    glVertex3f(0.7, -0.4, 0);
   glEnd();

}

void DrawForkLift(float x, float y, float yaw,float steer)
{
    pangolin::OpenGlMatrix Twc,Twc_T ,Twc_R;
    Twc.SetIdentity();
    Twc_T = Twc.Translate(x, y, 0.0);
    Twc_R =Twc.RotateZ(yaw);

    Twc_R(0,3) += Twc_T(0,3);
    Twc_R(1,3) += Twc_T(1,3);
    Twc_R(2,3) += Twc_T(2,3);

    glPushMatrix();
    glMultMatrixd(Twc_R.m);
    DrawPassiveWheel();
    DrawRearSteerWheel(steer);
    glPopMatrix();
}

int main(int argc, char *argv[])
{
    pangolin::CreateWindowAndBind("ForkLift Model", 1024, 768);
    glEnable(GL_DEPTH_TEST);


    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 368, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, 0.0, 12, 0, 0, 0, 0.0, 1.0, 0.0));

    pangolin::View& d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                                                     .SetHandler(new pangolin::Handler3D(s_cam));

    float b = 1.3;

    float v = 0.3;
    float theta =3.14/2;

    float DT = 0.1;
    float x = 0;
    float y = 0;
    float z = 0;
    float x1 =0;
    float y1 = 0;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        drawGrid(10);


        double w = v*sin(theta)/b;
        z = z+(w*DT);
        x1  += v*cos(z)*DT;
       y1  += v*sin(z)*DT ;
        v = v+0.01*DT;
       // x = x-(b/tan(theta))*sin(z) + (b/tan(theta))*sin(z+(v*DT*sin(theta))/b);
        //y = y +(b/tan(theta))*cos(z) - (b/tan(theta))*cos(z + (v*DT*sin(theta))/b);

        DrawForkLift(x1,y1, z,theta);
       // DrawForkLift(x,y, z,theta);
        cout<<x1<<"=="<<x<<endl;
         pangolin::FinishFrame();
    }
}
#endif
