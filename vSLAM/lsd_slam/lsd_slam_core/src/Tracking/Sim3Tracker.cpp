/**
* 相似变换 Sim(3) 匹配变换跟踪
* 主要用于　闭环检测和全局优化
* 每一个候选帧与检测的关键帧做双向的Sim3跟踪，如果都成功，
* 并且两个sim(3)的马氏距离足够小，则认为是检测处闭环，
* 并且在位姿图中构建sim(3)的边。
* 
* 误差方程：
* 
* E = SUM((I(p)-I'(p'))^2+(d(p)-d'(p'))^2)
* 除了类似　SE3的光度误差　多了一项　逆深度误差
* 
* 求解sim3时也使用了和 求解se3是一样的算法
* 迭代变权重列文伯格马尔夸克LM算法
* 
* 这里对误差的处理差不多，都需要进行误差加权后方差归一化
* 误差　 rp   rd
* 加权    Wh
* 方差归一化　σp  σd
* 偏导数　Jp  Jd
* 
* 得到 求解相似变换sim3的线性方程
* (Wh/σp^2 * Jp转置 * Jp +  Wh/σd^2 * Jd转置 * Jd) * dsim3 = - (Wh*rp/σp^2 * Jp +  Wh*rd/σd^2 * Jd)
* 
* A * dsim3 = b
*　使用LDLT分解求dsim3 
* 然后将dsim3 指数映射后　对SIM3进行更新
*
* https://blog.csdn.net/kokerf/article/details/78006743
* 
*/

#include "Sim3Tracker.h"
#include <opencv2/highgui/highgui.hpp>
#include "DataStructures/Frame.h"
#include "Tracking/TrackingReference.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "Tracking/least_squares.h"

namespace lsd_slam
{


#if defined(ENABLE_NEON)
	#define callOptimized(function, arguments) function##NEON arguments
#else
	#if defined(ENABLE_SSE)
		#define callOptimized(function, arguments) (USESSE ? function##SSE arguments : function arguments)
	#else
		#define callOptimized(function, arguments) function arguments
	#endif
#endif



Sim3Tracker::Sim3Tracker(int w, int h, Eigen::Matrix3f K)
{
	width = w;
	height = h;

	this->K = K;
	fx = K(0,0);
	fy = K(1,1);
	cx = K(0,2);
	cy = K(1,2);

	settings = DenseDepthTrackerSettings();


	KInv = K.inverse();
	fxi = KInv(0,0);
	fyi = KInv(1,1);
	cxi = KInv(0,2);
	cyi = KInv(1,2);


	buf_warped_residual = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_weights = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dx = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dy = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_x = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_y = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_z = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_residual_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_p = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_weight_Huber = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_VarP = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_VarD = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_warped_size = 0;

	debugImageWeights = cv::Mat(height,width,CV_8UC3);
	debugImageResiduals = cv::Mat(height,width,CV_8UC3);
	debugImageSecondFrame = cv::Mat(height,width,CV_8UC3);
	debugImageOldImageWarped = cv::Mat(height,width,CV_8UC3);
	debugImageOldImageSource = cv::Mat(height,width,CV_8UC3);
	debugImageExternalWeights = cv::Mat(height,width,CV_8UC3);
	debugImageDepthResiduals = cv::Mat(height,width,CV_8UC3);
	debugImageScaleEstimation = cv::Mat(height,width,CV_8UC3);

	debugImageHuberWeight = cv::Mat(height,width,CV_8UC3);
	debugImageWeightD = cv::Mat(height,width,CV_8UC3);
	debugImageWeightP = cv::Mat(height,width,CV_8UC3);
	debugImageWeightedResP = cv::Mat(height,width,CV_8UC3);
	debugImageWeightedResD = cv::Mat(height,width,CV_8UC3);

	
	lastResidual = 0;
	iterationNumber = 0;
	lastDepthResidual = lastPhotometricResidual = lastDepthResidualUnweighted = lastPhotometricResidualUnweighted = lastResidualUnweighted = 0;
	pointUsage = 0;

}

Sim3Tracker::~Sim3Tracker()
{
	debugImageResiduals.release();
	debugImageWeights.release();
	debugImageSecondFrame.release();
	debugImageOldImageSource.release();
	debugImageOldImageWarped.release();
	debugImageExternalWeights.release();
	debugImageDepthResiduals.release();
	debugImageScaleEstimation.release();

	debugImageHuberWeight.release();
	debugImageWeightD.release();
	debugImageWeightP.release();
	debugImageWeightedResP.release();
	debugImageWeightedResD.release();


	Eigen::internal::aligned_free((void*)buf_warped_residual);
	Eigen::internal::aligned_free((void*)buf_warped_weights);
	Eigen::internal::aligned_free((void*)buf_warped_dx);
	Eigen::internal::aligned_free((void*)buf_warped_dy);
	Eigen::internal::aligned_free((void*)buf_warped_x);
	Eigen::internal::aligned_free((void*)buf_warped_y);
	Eigen::internal::aligned_free((void*)buf_warped_z);

	Eigen::internal::aligned_free((void*)buf_d);
	Eigen::internal::aligned_free((void*)buf_residual_d);
	Eigen::internal::aligned_free((void*)buf_idepthVar);
	Eigen::internal::aligned_free((void*)buf_warped_idepthVar);
	Eigen::internal::aligned_free((void*)buf_weight_p);
	Eigen::internal::aligned_free((void*)buf_weight_d);

	Eigen::internal::aligned_free((void*)buf_weight_Huber);
	Eigen::internal::aligned_free((void*)buf_weight_VarP);
	Eigen::internal::aligned_free((void*)buf_weight_VarD);
}


Sim3 Sim3Tracker::trackFrameSim3(
		TrackingReference* reference,
		Frame* frame,
		const Sim3& frameToReference_initialEstimate,
		int startLevel, int finalLevel)
{
	boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();

	diverged = false;


	affineEstimation_a = 1; affineEstimation_b = 0;


	// ============ track frame ============
    Sim3 referenceToFrame = frameToReference_initialEstimate.inverse();
	NormalEquationsLeastSquares7 ls7;


	int numCalcResidualCalls[PYRAMID_LEVELS];
	int numCalcWarpUpdateCalls[PYRAMID_LEVELS];

	Sim3ResidualStruct finalResidual;

	bool warp_update_up_to_date = false;

	for(int lvl=startLevel;lvl >= finalLevel;lvl--)
	{
		numCalcResidualCalls[lvl] = 0;
		numCalcWarpUpdateCalls[lvl] = 0;

		if(settings.maxItsPerLvl[lvl] == 0)
			continue;

		reference->makePointCloud(lvl);

		// evaluate baseline-residual.
		callOptimized(calcSim3Buffers, (reference, frame, referenceToFrame, lvl));
		if(buf_warped_size < 0.5 * MIN_GOODPERALL_PIXEL_ABSMIN * (width>>lvl)*(height>>lvl) || buf_warped_size < 10)
		{
			diverged = true;
			return Sim3();
		}

		Sim3ResidualStruct lastErr = callOptimized(calcSim3WeightsAndResidual,(referenceToFrame));
		if(plotSim3TrackingIterationInfo) callOptimized(calcSim3Buffers,(reference, frame, referenceToFrame, lvl, true));
		numCalcResidualCalls[lvl]++;

		if(useAffineLightningEstimation)
		{
			affineEstimation_a = affineEstimation_a_lastIt;
			affineEstimation_b = affineEstimation_b_lastIt;
		}

		float LM_lambda = settings.lambdaInitial[lvl];

		warp_update_up_to_date = false;
		for(int iteration=0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
		{

			// calculate LS System, result is saved in ls.
			callOptimized(calcSim3LGS,(ls7));
			warp_update_up_to_date = true;
			numCalcWarpUpdateCalls[lvl]++;

			iterationNumber = iteration;


			int incTry=0;
			while(true)
			{
				// solve LS system with current lambda
				Vector7 b = - ls7.b / ls7.num_constraints;
				Matrix7x7 A = ls7.A / ls7.num_constraints;
				for(int i=0;i<7;i++) A(i,i) *= 1+LM_lambda;
				Vector7 inc = A.ldlt().solve(b);// LDLT分解求解线性方程组
				incTry++;

				float absInc = inc.dot(inc);
				if(!(absInc >= 0 && absInc < 1))
				{
					// ERROR tracking diverged.
					lastSim3Hessian.setZero();
					return Sim3();
				}

				// apply increment. pretty sure this way round is correct, but hard to test.
				Sim3 new_referenceToFrame =Sim3::exp(inc.cast<sophusType>()) * referenceToFrame;
				//Sim3 new_referenceToFrame = referenceToFrame * Sim3::exp((inc));


				// re-evaluate residual
				callOptimized(calcSim3Buffers,(reference, frame, new_referenceToFrame, lvl));
				if(buf_warped_size < 0.5 * MIN_GOODPERALL_PIXEL_ABSMIN * (width>>lvl)*(height>>lvl) || buf_warped_size < 10)
				{
					diverged = true;
					return Sim3();
				}

				Sim3ResidualStruct error = callOptimized(calcSim3WeightsAndResidual,(new_referenceToFrame));
				if(plotSim3TrackingIterationInfo) callOptimized(calcSim3Buffers,(reference, frame, new_referenceToFrame, lvl, true));
				numCalcResidualCalls[lvl]++;


				// accept inc?
				if(error.mean < lastErr.mean)
				{
					// accept inc
					referenceToFrame = new_referenceToFrame;
					warp_update_up_to_date = false;

					if(useAffineLightningEstimation)
					{
						affineEstimation_a = affineEstimation_a_lastIt;
						affineEstimation_b = affineEstimation_b_lastIt;
					}

					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						// debug output
						printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);

						printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
								referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
								referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
								referenceToFrame.log()[6]);
					}

					// converged?
					if(error.mean / lastErr.mean > settings.convergenceEps[lvl])
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
					}

					finalResidual = lastErr = error;

					if(LM_lambda <= 0.2)
						LM_lambda = 0;
					else
						LM_lambda *= settings.lambdaSuccessFac;

					break;
				}
				else
				{
					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						printf("(%d-%d): REJECTED increment of %f with lambda %.1f, (residual: %f -> %f)\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);
					}

					if(!(inc.dot(inc) > settings.stepSizeMin[lvl]))
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (stepsize too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
						break;
					}

					if(LM_lambda == 0)
						LM_lambda = 0.2;
					else
						LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
				}
			}
		}
	}



	if(enablePrintDebugInfo && printTrackingIterationInfo)
	{
		printf("Tracking: ");
			for(int lvl=PYRAMID_LEVELS-1;lvl >= 0;lvl--)
			{
				printf("lvl %d: %d (%d); ",
					lvl,
					numCalcResidualCalls[lvl],
					numCalcWarpUpdateCalls[lvl]);
			}

		printf("\n");


		printf("pOld = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
				frameToReference_initialEstimate.inverse().log()[0],frameToReference_initialEstimate.inverse().log()[1],frameToReference_initialEstimate.inverse().log()[2],
				frameToReference_initialEstimate.inverse().log()[3],frameToReference_initialEstimate.inverse().log()[4],frameToReference_initialEstimate.inverse().log()[5],
				frameToReference_initialEstimate.inverse().log()[6]);
		printf("pNew = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
				referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
				referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
				referenceToFrame.log()[6]);
		printf("final res mean: %f meanD %f, meanP %f\n", finalResidual.mean, finalResidual.meanD, finalResidual.meanP);
	}


	// Make sure that there is a warp update at the final position to get the correct information matrix
	if (!warp_update_up_to_date)
	{
		reference->makePointCloud(finalLevel);
		callOptimized(calcSim3Buffers,(reference, frame, referenceToFrame, finalLevel));
	    finalResidual = callOptimized(calcSim3WeightsAndResidual,(referenceToFrame));
	    callOptimized(calcSim3LGS,(ls7));
	}

	lastSim3Hessian = ls7.A;


	if(referenceToFrame.scale() <= 0 )
	{
		diverged = true;
		return Sim3();
	}

	lastResidual = finalResidual.mean;
	lastDepthResidual = finalResidual.meanD;
	lastPhotometricResidual = finalResidual.meanP;


	return referenceToFrame.inverse();
}




#if defined(ENABLE_SSE)
void Sim3Tracker::calcSim3BuffersSSE(
		const TrackingReference* reference,
		Frame* frame,
		const Sim3& referenceToFrame,
		int level, bool plotWeights)
{
	calcSim3Buffers(
			reference,
			frame,
			referenceToFrame,
			level, plotWeights);
}
#endif

#if defined(ENABLE_NEON)
void Sim3Tracker::calcSim3BuffersNEON(
		const TrackingReference* reference,
		Frame* frame,
		const Sim3& referenceToFrame,
		int level, bool plotWeights)
{
	calcSim3Buffers(
			reference,
			frame,
			referenceToFrame,
			level, plotWeights);
}
#endif


void Sim3Tracker::calcSim3Buffers(
		const TrackingReference* reference,
		Frame* frame,
		const Sim3& referenceToFrame,
		int level, bool plotWeights)
{
	if(plotSim3TrackingIterationInfo)
	{
		cv::Vec3b col = cv::Vec3b(255,170,168);
		fillCvMat(&debugImageResiduals,col);
		fillCvMat(&debugImageOldImageSource,col);
		fillCvMat(&debugImageOldImageWarped,col);
		fillCvMat(&debugImageDepthResiduals,col);
	}
	if(plotWeights && plotSim3TrackingIterationInfo)
	{
		cv::Vec3b col = cv::Vec3b(255,170,168);
		fillCvMat(&debugImageHuberWeight,col);
		fillCvMat(&debugImageWeightD,col);
		fillCvMat(&debugImageWeightP,col);
		fillCvMat(&debugImageWeightedResP,col);
		fillCvMat(&debugImageWeightedResD,col);
	}

	// get static values
	int w = frame->width(level);
	int h = frame->height(level);
	Eigen::Matrix3f KLvl = frame->K(level);
	float fx_l = KLvl(0,0);
	float fy_l = KLvl(1,1);
	float cx_l = KLvl(0,2);
	float cy_l = KLvl(1,2);

	Eigen::Matrix3f rotMat = referenceToFrame.rxso3().matrix().cast<float>();
	Eigen::Matrix3f rotMatUnscaled = referenceToFrame.rotationMatrix().cast<float>();
	Eigen::Vector3f transVec = referenceToFrame.translation().cast<float>();

	// Calculate rotation around optical axis for rotating source frame gradients
	Eigen::Vector3f forwardVector(0, 0, -1);
	Eigen::Vector3f rotatedForwardVector = rotMatUnscaled * forwardVector;
	Eigen::Quaternionf shortestBackRotation;
	shortestBackRotation.setFromTwoVectors(rotatedForwardVector, forwardVector);
	Eigen::Matrix3f rollMat = shortestBackRotation.toRotationMatrix() * rotMatUnscaled;
	float xRoll0 = rollMat(0, 0);
	float xRoll1 = rollMat(0, 1);
	float yRoll0 = rollMat(1, 0);
	float yRoll1 = rollMat(1, 1);


	const Eigen::Vector3f* refPoint_max = reference->posData[level] + reference->numData[level];
	const Eigen::Vector3f* refPoint = reference->posData[level];
	const Eigen::Vector2f* refColVar = reference->colorAndVarData[level];
	const Eigen::Vector2f* refGrad = reference->gradData[level];

	const float* 			frame_idepth = frame->idepth(level);
	const float* 			frame_idepthVar = frame->idepthVar(level);
	const Eigen::Vector4f* 	frame_intensityAndGradients = frame->gradients(level);


	float sxx=0,syy=0,sx=0,sy=0,sw=0;

	float usageCount = 0;

	int idx=0;
	for(;refPoint<refPoint_max; refPoint++, refGrad++, refColVar++)
	{
		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;

		// step 1a: coordinates have to be in image:
		// (inverse test to exclude NANs)
		if(!(u_new > 1 && v_new > 1 && u_new < w-2 && v_new < h-2))
			continue;

		*(buf_warped_x+idx) = Wxp(0);
		*(buf_warped_y+idx) = Wxp(1);
		*(buf_warped_z+idx) = Wxp(2);

		Eigen::Vector3f resInterp = getInterpolatedElement43(frame_intensityAndGradients, u_new, v_new, w);


		// save values
#if USE_ESM_TRACKING == 1
		// get rotated gradient of point
		float rotatedGradX = xRoll0 * (*refGrad)[0] + xRoll1 * (*refGrad)[1];
		float rotatedGradY = yRoll0 * (*refGrad)[0] + yRoll1 * (*refGrad)[1];

		*(buf_warped_dx+idx) = fx_l * 0.5f * (resInterp[0] + rotatedGradX);
		*(buf_warped_dy+idx) = fy_l * 0.5f * (resInterp[1] + rotatedGradY);
#else
		*(buf_warped_dx+idx) = fx_l * resInterp[0];
		*(buf_warped_dy+idx) = fy_l * resInterp[1];
#endif


		float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
		float c2 = resInterp[2];
		float residual_p = c1 - c2;

		float weight = fabsf(residual_p) < 2.0f ? 1 : 2.0f / fabsf(residual_p);
		sxx += c1*c1*weight;
		syy += c2*c2*weight;
		sx += c1*weight;
		sy += c2*weight;
		sw += weight;


		*(buf_warped_residual+idx) = residual_p;
		*(buf_idepthVar+idx) = (*refColVar)[1];


		// new (only for Sim3):
		int idx_rounded = (int)(u_new+0.5f) + w*(int)(v_new+0.5f);
		float var_frameDepth = frame_idepthVar[idx_rounded];
		float ref_idepth = 1.0f / Wxp[2];
		*(buf_d+idx) = 1.0f / (*refPoint)[2];
		if(var_frameDepth > 0)
		{
			float residual_d = ref_idepth - frame_idepth[idx_rounded];
			*(buf_residual_d+idx) = residual_d;
			*(buf_warped_idepthVar+idx) = var_frameDepth;
		}
		else
		{
			*(buf_residual_d+idx) = -1;
			*(buf_warped_idepthVar+idx) = -1;
		}


		// DEBUG STUFF
		if(plotSim3TrackingIterationInfo)
		{
			// for debug plot only: find x,y again.
			// horribly inefficient, but who cares at this point...
			Eigen::Vector3f point = KLvl * (*refPoint);
			int x = point[0] / point[2] + 0.5f;
			int y = point[1] / point[2] + 0.5f;

			setPixelInCvMat(&debugImageOldImageSource,getGrayCvPixel((float)resInterp[2]),u_new+0.5,v_new+0.5,(width/w));
			setPixelInCvMat(&debugImageOldImageWarped,getGrayCvPixel((float)resInterp[2]),x,y,(width/w));
			setPixelInCvMat(&debugImageResiduals,getGrayCvPixel(residual_p+128),x,y,(width/w));

			if(*(buf_warped_idepthVar+idx) >= 0)
			{
				setPixelInCvMat(&debugImageDepthResiduals,getGrayCvPixel(128 + 800 * *(buf_residual_d+idx)),x,y,(width/w));

				if(plotWeights)
				{
					setPixelInCvMat(&debugImageWeightD,getGrayCvPixel(255 * (1/60.0f) * sqrtf(*(buf_weight_VarD+idx))),x,y,(width/w));
					setPixelInCvMat(&debugImageWeightedResD,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarD+idx)) * *(buf_residual_d+idx)),x,y,(width/w));
				}
			}


			if(plotWeights)
			{
				setPixelInCvMat(&debugImageWeightP,getGrayCvPixel(255 * 4 * sqrtf(*(buf_weight_VarP+idx))),x,y,(width/w));
				setPixelInCvMat(&debugImageHuberWeight,getGrayCvPixel(255 * *(buf_weight_Huber+idx)),x,y,(width/w));
				setPixelInCvMat(&debugImageWeightedResP,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarP+idx)) * *(buf_warped_residual+idx)),x,y,(width/w));
			}
		}

		idx++;

		float depthChange = (*refPoint)[2] / Wxp[2];
		usageCount += depthChange < 1 ? depthChange : 1;
	}
	buf_warped_size = idx;


	pointUsage = usageCount / (float)reference->numData[level];

	affineEstimation_a_lastIt = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
	affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx)/sw;



	if(plotSim3TrackingIterationInfo)
	{
		Util::displayImage( "P Residuals", debugImageResiduals );
		Util::displayImage( "D Residuals", debugImageDepthResiduals );

		if(plotWeights)
		{
			Util::displayImage( "Huber Weights", debugImageHuberWeight );
			Util::displayImage( "DV Weights", debugImageWeightD );
			Util::displayImage( "IV Weights", debugImageWeightP );
			Util::displayImage( "WP Res", debugImageWeightedResP );
			Util::displayImage( "WD Res", debugImageWeightedResD );
		}
	}

}


#if defined(ENABLE_SSE)
Sim3ResidualStruct Sim3Tracker::calcSim3WeightsAndResidualSSE(
		const Sim3& referenceToFrame)
{

	const __m128 txs = _mm_set1_ps((float)(referenceToFrame.translation()[0]));
	const __m128 tys = _mm_set1_ps((float)(referenceToFrame.translation()[1]));
	const __m128 tzs = _mm_set1_ps((float)(referenceToFrame.translation()[2]));

	const __m128 zeros = _mm_set1_ps(0.0f);
	const __m128 ones = _mm_set1_ps(1.0f);


	const __m128 depthVarFacs = _mm_set1_ps((float)settings.var_weight);// float depthVarFac = var_weight;	// the depth var is over-confident. this is a constant multiplier to remedy that.... HACK
	const __m128 sigma_i2s = _mm_set1_ps((float)cameraPixelNoise2);


	const __m128 huber_ress = _mm_set1_ps((float)(settings.huber_d));

	__m128 sumResP = zeros;
	__m128 sumResD = zeros;
	__m128 numTermsD = zeros;



	Sim3ResidualStruct sumRes;
	memset(&sumRes, 0, sizeof(Sim3ResidualStruct));


	for(int i=0;i<buf_warped_size-3;i+=4)
	{

		// calc dw/dd:
		 //float g0 = (tx * pz - tz * px) / (pz*pz*d);
		__m128 pzs = _mm_load_ps(buf_warped_z+i);	// z'
		__m128 pz2ds = _mm_rcp_ps(_mm_mul_ps(_mm_mul_ps(pzs, pzs), _mm_load_ps(buf_d+i)));  // 1 / (z' * z' * d)
		__m128 g0s = _mm_sub_ps(_mm_mul_ps(pzs, txs), _mm_mul_ps(_mm_load_ps(buf_warped_x+i), tzs));
		g0s = _mm_mul_ps(g0s,pz2ds);

		 //float g1 = (ty * pz - tz * py) / (pz*pz*d);
		__m128 g1s = _mm_sub_ps(_mm_mul_ps(pzs, tys), _mm_mul_ps(_mm_load_ps(buf_warped_y+i), tzs));
		g1s = _mm_mul_ps(g1s,pz2ds);

		 //float g2 = (pz - tz) / (pz*pz*d);
		__m128 g2s = _mm_sub_ps(pzs, tzs);
		g2s = _mm_mul_ps(g2s,pz2ds);

		// calc w_p
		 // float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		__m128 drpdds = _mm_add_ps(
				_mm_mul_ps(g0s, _mm_load_ps(buf_warped_dx+i)),
				_mm_mul_ps(g1s, _mm_load_ps(buf_warped_dy+i)));

		 //float w_p = 1.0f / (sigma_i2 + s * drpdd * drpdd);
		__m128 w_ps = _mm_rcp_ps(_mm_add_ps(sigma_i2s,
				_mm_mul_ps(drpdds,
						_mm_mul_ps(drpdds,
								_mm_mul_ps(depthVarFacs,
										_mm_load_ps(buf_idepthVar+i))))));


		//float w_d = 1.0f / (sv + g2*g2*s);
		__m128 w_ds = _mm_rcp_ps(_mm_add_ps(_mm_load_ps(buf_warped_idepthVar+i),
				_mm_mul_ps(g2s,
						_mm_mul_ps(g2s,
								_mm_mul_ps(depthVarFacs,
										_mm_load_ps(buf_idepthVar+i))))));

		//float weighted_rp = fabs(rp*sqrtf(w_p));
		__m128 weighted_rps = _mm_mul_ps(_mm_load_ps(buf_warped_residual+i),
				_mm_sqrt_ps(w_ps));
		weighted_rps = _mm_max_ps(weighted_rps, _mm_sub_ps(zeros,weighted_rps));

		//float weighted_rd = fabs(rd*sqrtf(w_d));
		__m128 weighted_rds = _mm_mul_ps(_mm_load_ps(buf_residual_d+i),
				_mm_sqrt_ps(w_ds));
		weighted_rds = _mm_max_ps(weighted_rds, _mm_sub_ps(zeros,weighted_rds));

		// depthValid = sv > 0
		__m128 depthValid = _mm_cmplt_ps(zeros, _mm_load_ps(buf_warped_idepthVar+i));	// bitmask 0xFFFFFFFF for idepth valid, 0x000000 otherwise


		// float weighted_abs_res = sv > 0 ? weighted_rd+weighted_rp : weighted_rp;
		__m128 weighted_abs_ress = _mm_add_ps(_mm_and_ps(weighted_rds,depthValid), weighted_rps);

		//float wh = fabs(weighted_abs_res < huber_res ? 1 : huber_res / weighted_abs_res);
		__m128 whs = _mm_cmplt_ps(weighted_abs_ress, huber_ress);	// bitmask 0xFFFFFFFF for 1, 0x000000 for huber_res_ponly / weighted_rp
		whs = _mm_or_ps(
				_mm_and_ps(whs, ones),
				_mm_andnot_ps(whs, _mm_mul_ps(huber_ress, _mm_rcp_ps(weighted_abs_ress))));


		if(i+3 < buf_warped_size)
		{
			//if(sv > 0) sumRes.numTermsD++;
			numTermsD = _mm_add_ps(numTermsD,
					_mm_and_ps(depthValid, ones));

			//if(sv > 0) sumRes.sumResD += wh * w_d * rd*rd;
			sumResD = _mm_add_ps(sumResD,
					_mm_and_ps(depthValid, _mm_mul_ps(whs, _mm_mul_ps(weighted_rds, weighted_rds))));

			// sumRes.sumResP += wh * w_p * rp*rp;
			sumResP = _mm_add_ps(sumResP,
					_mm_mul_ps(whs, _mm_mul_ps(weighted_rps, weighted_rps)));
		}

		//*(buf_weight_p+i) = wh * w_p;
		_mm_store_ps(buf_weight_p+i, _mm_mul_ps(whs, w_ps) );

		//if(sv > 0) *(buf_weight_d+i) = wh * w_d; else *(buf_weight_d+i) = 0;
		_mm_store_ps(buf_weight_d+i, _mm_and_ps(depthValid, _mm_mul_ps(whs, w_ds)) );

	}
	sumRes.sumResP = SSEE(sumResP,0) + SSEE(sumResP,1) + SSEE(sumResP,2) + SSEE(sumResP,3);
	sumRes.numTermsP = (buf_warped_size >> 2) << 2;

	sumRes.sumResD = SSEE(sumResD,0) + SSEE(sumResD,1) + SSEE(sumResD,2) + SSEE(sumResD,3);
	sumRes.numTermsD = SSEE(numTermsD,0) + SSEE(numTermsD,1) + SSEE(numTermsD,2) + SSEE(numTermsD,3);

	sumRes.mean = (sumRes.sumResD + sumRes.sumResP) / (sumRes.numTermsD + sumRes.numTermsP);
	sumRes.meanD = (sumRes.sumResD) / (sumRes.numTermsD);
	sumRes.meanP = (sumRes.sumResP) / (sumRes.numTermsP);

	return sumRes;
}
#endif

#if defined(ENABLE_NEON)
Sim3ResidualStruct Sim3Tracker::calcSim3WeightsAndResidualNEON(
		const Sim3& referenceToFrame)
{
	return calcSim3WeightsAndResidual(
			referenceToFrame);
}
#endif


Sim3ResidualStruct Sim3Tracker::calcSim3WeightsAndResidual(
		const Sim3& referenceToFrame)
{
	float tx = referenceToFrame.translation()[0];
	float ty = referenceToFrame.translation()[1];
	float tz = referenceToFrame.translation()[2];

	Sim3ResidualStruct sumRes;
	memset(&sumRes, 0, sizeof(Sim3ResidualStruct));


	float sum_rd=0, sum_rp=0, sum_wrd=0, sum_wrp=0, sum_wp=0, sum_wd=0, sum_num_d=0, sum_num_p=0;

	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);	// x'
		float py = *(buf_warped_y+i);	// y'
		float pz = *(buf_warped_z+i);	// z'

		float d = *(buf_d+i);	// d

		float rp = *(buf_warped_residual+i); // r_p
		float rd = *(buf_residual_d+i);	 // r_d

		float gx = *(buf_warped_dx+i);	// \delta_x I
		float gy = *(buf_warped_dy+i);  // \delta_y I

		float s = settings.var_weight * *(buf_idepthVar+i);	// \sigma_d^2
		float sv = settings.var_weight * *(buf_warped_idepthVar+i);	// \sigma_d^2'


		// calc dw/dd (first 2 components):
		float g0 = (tx * pz - tz * px) / (pz*pz*d);
		float g1 = (ty * pz - tz * py) / (pz*pz*d);
		float g2 = (pz - tz) / (pz*pz*d);

		// calc w_p
		float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		float w_p = 1.0f / (cameraPixelNoise2 + s * drpdd * drpdd);

		float w_d = 1.0f / (sv + g2*g2*s);

		float weighted_rd = fabs(rd*sqrtf(w_d));
		float weighted_rp = fabs(rp*sqrtf(w_p));


		float weighted_abs_res = sv > 0 ? weighted_rd+weighted_rp : weighted_rp;
		float wh = fabs(weighted_abs_res < settings.huber_d ? 1 : settings.huber_d / weighted_abs_res);

		if(sv > 0)
		{
			sumRes.sumResD += wh * w_d * rd*rd;
			sumRes.numTermsD++;
		}

		sumRes.sumResP += wh * w_p * rp*rp;
		sumRes.numTermsP++;


		if(plotSim3TrackingIterationInfo)
		{
			// for debug
			*(buf_weight_Huber+i) = wh;
			*(buf_weight_VarP+i) = w_p;
			*(buf_weight_VarD+i) = sv > 0 ? w_d : 0;


			sum_rp += fabs(rp);
			sum_wrp += fabs(weighted_rp);
			sum_wp += sqrtf(w_p);
			sum_num_p++;

			if(sv > 0)
			{
				sum_rd += fabs(weighted_rd);
				sum_wrd += fabs(rd);
				sum_wd += sqrtf(w_d);
				sum_num_d++;
			}
		}

		*(buf_weight_p+i) = wh * w_p;

		if(sv > 0)
			*(buf_weight_d+i) = wh * w_d;
		else
			*(buf_weight_d+i) = 0;

	}

	sumRes.mean = (sumRes.sumResD + sumRes.sumResP) / (sumRes.numTermsD + sumRes.numTermsP);
	sumRes.meanD = (sumRes.sumResD) / (sumRes.numTermsD);
	sumRes.meanP = (sumRes.sumResP) / (sumRes.numTermsP);

	if(plotSim3TrackingIterationInfo)
	{
		printf("rd %f, rp %f, wrd %f, wrp %f, wd %f, wp %f\n ",
				sum_rd/sum_num_d,
				sum_rp/sum_num_p,
				sum_wrd/sum_num_d,
				sum_wrp/sum_num_p,
				sum_wd/sum_num_d,
				sum_wp/sum_num_p);
	}
	return sumRes;
}



#if defined(ENABLE_SSE)
void Sim3Tracker::calcSim3LGSSSE(NormalEquationsLeastSquares7 &ls7)
{
	NormalEquationsLeastSquares4 ls4;
	NormalEquationsLeastSquares ls6;
	ls6.initialize(width*height);
	ls4.initialize(width*height);

	const __m128 zeros = _mm_set1_ps(0.0f);

	for(int i=0;i<buf_warped_size-3;i+=4)
	{
		Vector6 v1,v2,v3,v4;
		Vector4 v41,v42,v43,v44;
		__m128 val1, val2, val3, val4;

		// redefine pz
		__m128 pz = _mm_load_ps(buf_warped_z+i);
		pz = _mm_rcp_ps(pz);						// pz := 1/z

		//v4[3] = z;
		v41[3] = SSEE(pz,0);
		v42[3] = SSEE(pz,1);
		v43[3] = SSEE(pz,2);
		v44[3] = SSEE(pz,3);

		__m128 gx = _mm_load_ps(buf_warped_dx+i);
		val1 = _mm_mul_ps(pz, gx);			// gx / z => SET [0]
		//v[0] = z*gx;
		v1[0] = SSEE(val1,0);
		v2[0] = SSEE(val1,1);
		v3[0] = SSEE(val1,2);
		v4[0] = SSEE(val1,3);




		__m128 gy = _mm_load_ps(buf_warped_dy+i);
		val1 = _mm_mul_ps(pz, gy);					// gy / z => SET [1]
		//v[1] = z*gy;
		v1[1] = SSEE(val1,0);
		v2[1] = SSEE(val1,1);
		v3[1] = SSEE(val1,2);
		v4[1] = SSEE(val1,3);


		__m128 px = _mm_load_ps(buf_warped_x+i);
		val1 = _mm_mul_ps(px, gy);
		val1 = _mm_mul_ps(val1, pz);	//  px * gy * z
		__m128 py = _mm_load_ps(buf_warped_y+i);
		val2 = _mm_mul_ps(py, gx);
		val2 = _mm_mul_ps(val2, pz);	//  py * gx * z
		val1 = _mm_sub_ps(val1, val2);  // px * gy * z - py * gx * z => SET [5]
		//v[5] = -py * z * gx +  px * z * gy;
		v1[5] = SSEE(val1,0);
		v2[5] = SSEE(val1,1);
		v3[5] = SSEE(val1,2);
		v4[5] = SSEE(val1,3);


		// redefine pz
		pz = _mm_mul_ps(pz,pz); 		// pz := 1/(z*z)

		//v4[0] = z_sqr;
		v41[0] = SSEE(pz,0);
		v42[0] = SSEE(pz,1);
		v43[0] = SSEE(pz,2);
		v44[0] = SSEE(pz,3);

		//v4[1] = z_sqr * py;
		__m128 pypz = _mm_mul_ps(pz, py);
		v41[1] = SSEE(pypz,0);
		v42[1] = SSEE(pypz,1);
		v43[1] = SSEE(pypz,2);
		v44[1] = SSEE(pypz,3);

		//v4[2] = -z_sqr * px;
		__m128 mpxpz = _mm_sub_ps(zeros,_mm_mul_ps(pz, px));
		v41[2] = SSEE(mpxpz,0);
		v42[2] = SSEE(mpxpz,1);
		v43[2] = SSEE(mpxpz,2);
		v44[2] = SSEE(mpxpz,3);



		// will use these for the following calculations a lot.
		val1 = _mm_mul_ps(px, gx);
		val1 = _mm_mul_ps(val1, pz);		// px * z_sqr * gx
		val2 = _mm_mul_ps(py, gy);
		val2 = _mm_mul_ps(val2, pz);		// py * z_sqr * gy


		val3 = _mm_add_ps(val1, val2);
		val3 = _mm_sub_ps(zeros,val3);	//-px * z_sqr * gx -py * z_sqr * gy
		//v[2] = -px * z_sqr * gx -py * z_sqr * gy;	=> SET [2]
		v1[2] = SSEE(val3,0);
		v2[2] = SSEE(val3,1);
		v3[2] = SSEE(val3,2);
		v4[2] = SSEE(val3,3);


		val3 = _mm_mul_ps(val1, py); // px * z_sqr * gx * py
		val4 = _mm_add_ps(gy, val3); // gy + px * z_sqr * gx * py
		val3 = _mm_mul_ps(val2, py); // py * py * z_sqr * gy
		val4 = _mm_add_ps(val3, val4); // gy + px * z_sqr * gx * py + py * py * z_sqr * gy
		val4 = _mm_sub_ps(zeros,val4); //val4 = -val4.
		//v[3] = -px * py * z_sqr * gx +
		//       -py * py * z_sqr * gy +
		//       -gy;		=> SET [3]
		v1[3] = SSEE(val4,0);
		v2[3] = SSEE(val4,1);
		v3[3] = SSEE(val4,2);
		v4[3] = SSEE(val4,3);


		val3 = _mm_mul_ps(val1, px); // px * px * z_sqr * gx
		val4 = _mm_add_ps(gx, val3); // gx + px * px * z_sqr * gx
		val3 = _mm_mul_ps(val2, px); // px * py * z_sqr * gy
		val4 = _mm_add_ps(val4, val3); // gx + px * px * z_sqr * gx + px * py * z_sqr * gy
		//v[4] = px * px * z_sqr * gx +
		//	   px * py * z_sqr * gy +
		//	   gx;				=> SET [4]
		v1[4] = SSEE(val4,0);
		v2[4] = SSEE(val4,1);
		v3[4] = SSEE(val4,2);
		v4[4] = SSEE(val4,3);


		// step 6: integrate into A and b:
		ls6.update(v1, *(buf_warped_residual+i+0), *(buf_weight_p+i+0));
		ls4.update(v41, *(buf_residual_d+i+0), *(buf_weight_d+i+0));

		if(i+1>=buf_warped_size) break;
		ls6.update(v2, *(buf_warped_residual+i+1), *(buf_weight_p+i+1));
		ls4.update(v42, *(buf_residual_d+i+1), *(buf_weight_d+i+1));

		if(i+2>=buf_warped_size) break;
		ls6.update(v3, *(buf_warped_residual+i+2), *(buf_weight_p+i+2));
		ls4.update(v43, *(buf_residual_d+i+2), *(buf_weight_d+i+2));

		if(i+3>=buf_warped_size) break;
		ls6.update(v4, *(buf_warped_residual+i+3), *(buf_weight_p+i+3));
		ls4.update(v44, *(buf_residual_d+i+3), *(buf_weight_d+i+3));
	}

	ls4.finishNoDivide();
	ls6.finishNoDivide();


	ls7.initializeFrom(ls6, ls4);


}
#endif

#if defined(ENABLE_NEON)
void Sim3Tracker::calcSim3LGSNEON(NormalEquationsLeastSquares7 &ls7)
{
	calcSim3LGS(ls7);
}
#endif


void Sim3Tracker::calcSim3LGS(NormalEquationsLeastSquares7 &ls7)
{
	NormalEquationsLeastSquares4 ls4;
	NormalEquationsLeastSquares ls6;
	ls6.initialize(width*height);
	ls4.initialize(width*height);

	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);	// x'
		float py = *(buf_warped_y+i);	// y'
		float pz = *(buf_warped_z+i);	// z'

		float wp = *(buf_weight_p+i);	// wr/wp
		float wd = *(buf_weight_d+i);	// wr/wd

		float rp = *(buf_warped_residual+i); // r_p
		float rd = *(buf_residual_d+i);	 // r_d

		float gx = *(buf_warped_dx+i);	// \delta_x I
		float gy = *(buf_warped_dy+i);  // \delta_y I


		float z = 1.0f / pz;
		float z_sqr = 1.0f / (pz*pz);
		Vector6 v;
		Vector4 v4;
		v[0] = z*gx + 0;
		v[1] = 0 +         z*gy;
		v[2] = (-px * z_sqr) * gx +
			  (-py * z_sqr) * gy;
		v[3] = (-px * py * z_sqr) * gx +
			  (-(1.0 + py * py * z_sqr)) * gy;
		v[4] = (1.0 + px * px * z_sqr) * gx +
			  (px * py * z_sqr) * gy;
		v[5] = (-py * z) * gx +
			  (px * z) * gy;

		// new:
		v4[0] = z_sqr;
		v4[1] = z_sqr * py;
		v4[2] = -z_sqr * px;
		v4[3] = z;

		ls6.update(v, rp, wp);		// Jac = - v
		ls4.update(v4, rd, wd);	// Jac = v4

	}

	ls4.finishNoDivide();
	ls6.finishNoDivide();


	ls7.initializeFrom(ls6, ls4);

}

void Sim3Tracker::calcResidualAndBuffers_debugStart()
{
	if(plotTrackingIterationInfo || saveAllTrackingStagesInternal)
	{
		int other = saveAllTrackingStagesInternal ? 255 : 0;
		fillCvMat(&debugImageResiduals,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageExternalWeights,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageWeights,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageSource,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageWarped,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageScaleEstimation,cv::Vec3b(255,other,other));
		fillCvMat(&debugImageDepthResiduals,cv::Vec3b(other,other,255));
	}
}

void Sim3Tracker::calcResidualAndBuffers_debugFinish(int w)
{
	if(plotTrackingIterationInfo)
	{
		Util::displayImage( "Weights", debugImageWeights );
		Util::displayImage( "second_frame", debugImageSecondFrame );
		Util::displayImage( "Intensities of second_frame at transformed positions", debugImageOldImageSource );
		Util::displayImage( "Intensities of second_frame at pointcloud in first_frame", debugImageOldImageWarped );
		Util::displayImage( "Residuals", debugImageResiduals );
		Util::displayImage( "DepthVar Weights", debugImageExternalWeights );
		Util::displayImage( "Depth Residuals", debugImageDepthResiduals );

		// wait for key and handle it
		bool looping = true;
		while(looping)
		{
			int k = Util::waitKey(1);
			if(k == -1)
			{
				if(autoRunWithinFrame)
					break;
				else
					continue;
			}

			char key = k;
			if(key == ' ')
				looping = false;
			else
				handleKey(k);
		}
	}

	if(saveAllTrackingStagesInternal)
	{
		char charbuf[500];

		snprintf(charbuf,500,"save/%sresidual-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageResiduals);

		snprintf(charbuf,500,"save/%swarped-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageOldImageWarped);

		snprintf(charbuf,500,"save/%sweights-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageWeights);

		printf("saved three images for lvl %d, iteration %d\n",w,iterationNumber);
	}
}
}
