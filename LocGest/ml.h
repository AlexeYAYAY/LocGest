#pragma once
#include "FileWork.h"
#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <iomanip>

#include "math.h"
#define PI 3.14159265

using namespace cv;
using namespace ml;
using namespace std;
class ML
{
public:
	float AngleBetweenPoints2d(double Ax, double Ay, double Bx, double By, double Cx, double Cy, double Dx, double Dy)
	{
		double ABx = Bx - Ax;
		double ABy = By - Ay;
		double CDx = Dx - Cx;
		double CDy = Dy - Cy;
		double ABCD = ABx*CDx + ABy*CDy;
		double AB = sqrt(ABx*ABx + ABy*ABy);
		double CD = sqrt(CDx*CDx + CDy*CDy);
		double ratio = ABCD / (AB*CD);
		double angleRad = acos(ratio);
		double angleDeg = (angleRad * 180) / PI;
		if (angleDeg<0) {
			angleDeg = 180 + angleDeg;
		}
		return angleDeg;
	}
	vector<float> FeatureSelection(vector<float> data)
	{
		int select[54] = {
			0,0,0,1,1,1,1,1,1,1,
			1,1,1,1,1,1,1,1,1,1,
			1,1,1,1,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,
			0,0,0,0 };
		vector<float> vec;
		for (int i = 0; i < data.size(); i++)
		{
			if (select[i] == 1)
				vec.push_back(data[i]);

		}
		return vec;
	}
	vector<vector<double>> FeatureSelection(vector<vector<double>> data)
	{
		int select[54] = { 0,0,0,1,1,1,1,1,1,1,
			1,1,1,1,1,1,1,1,1,1,
			1,1,1,1,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,
			0,0,0,0 };
		vector<vector<double>> two_dim_vector;
		for (int i = 0; i < data.size(); i++)
		{
			vector<double> vec;
			for (int j = 0; j < data.at(0).size(); j++)
			{
				if (select[j] == 1)
				{
					vec.push_back(data[i][j]);
				}
			}
			vec.push_back(data[i][data.at(0).size() - 1]);
			two_dim_vector.push_back(vec);
		}
		return two_dim_vector;
	}
	vector<float> coordinatToAngle(vector<float> data)
	{
		vector<float> vec;
		vec.push_back(AngleBetweenPoints2d(data[0], data[1], data[3], data[4], data[6], data[7], data[3], data[4]));
		vec.push_back(AngleBetweenPoints2d(data[3], data[4], data[6], data[7], data[9], data[10], data[6], data[7]));
		vec.push_back(AngleBetweenPoints2d(data[0], data[1], data[12], data[13], data[15], data[16], data[12], data[13]));
		vec.push_back(AngleBetweenPoints2d(data[12], data[13], data[15], data[16], data[18], data[19], data[15], data[16]));
		return vec;
	}

	vector<vector<double>> coordinatToAngle(vector<vector<double>> data)
	{
		vector<vector<double>> two_dim_vector;
		for (int i = 0; i < data.size(); i++)
		{
			vector<double> vec;
			vec.push_back(AngleBetweenPoints2d(data[i][0], data[i][1], data[i][3], data[i][4], data[i][6], data[i][7], data[i][3], data[i][4]));
			vec.push_back(AngleBetweenPoints2d(data[i][3], data[i][4], data[i][6], data[i][7], data[i][9], data[i][10], data[i][6], data[i][7]));
			vec.push_back(AngleBetweenPoints2d(data[i][0], data[i][1], data[i][12], data[i][13], data[i][15], data[i][16], data[i][12], data[i][13]));
			vec.push_back(AngleBetweenPoints2d(data[i][12], data[i][13], data[i][15], data[i][16], data[i][18], data[i][19], data[i][15], data[i][16]));
			vec.push_back(data[i][data.at(0).size() - 1]);
			two_dim_vector.push_back(vec);
		}
		return two_dim_vector;
	}

	Ptr<ANN_MLP> Learning(vector<vector<double>> aa, std::string filename)
	{
		op::log("Starting OpenPoseWrapper...");
		vector<vector<double>> data = coordinatToAngle(FeatureSelection(aa));
		//op::log(aa[0]);	
		const int hiddenLayerSize = 4;
		int height = data.size();
		int width = data[0].size() - 1;
		//http://www.cplusplus.com/forum/general/41387/
		Mat inputTrainingData = Mat(height, width, CV_32F, cv::Scalar::all(0));
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++)
			{
				inputTrainingData.at<float>(i, j) = data[i][j];
			}
		Mat outputTrainingData = Mat(height, 1, CV_32F, cv::Scalar::all(0));
		for (int i = 0; i<height; i++)
		{
			outputTrainingData.at<float>(i, 0) = data[i][width];
		}
		Ptr<ANN_MLP> mlp = ANN_MLP::create();
		Mat layersSize = Mat(3, 1, CV_16U);
		layersSize.row(0) = Scalar(inputTrainingData.cols);
		layersSize.row(1) = Scalar(hiddenLayerSize);
		layersSize.row(2) = Scalar(outputTrainingData.cols);
		mlp->setLayerSizes(layersSize);
		mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
		TermCriteria termCrit = TermCriteria(
			TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
			100000000,
			0.000000000000000001
		);
		mlp->setTermCriteria(termCrit);
		mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
		Ptr<TrainData> trainingData = TrainData::create(
			inputTrainingData,
			SampleTypes::ROW_SAMPLE,
			outputTrainingData
		);
		std::cout<<"Training..."<<"\n";
		mlp->train(trainingData);
		mlp->save(filename.substr(0, filename.find(".csv")) + ".yml");
		std::cout << "Saved NN model " << filename.substr(0, filename.find(".csv")) + ".yml" <<"\n";
		for (int i = 0; i < inputTrainingData.rows; i++) {
			Mat sample = Mat(1, inputTrainingData.cols, CV_32F, cv::Scalar::all(0));
			for (int j = 0; j < sample.cols; j++)
			{
				sample.at<float>(j) = data[i][j];
			}
			Mat result;
			//op::log(sample);
			mlp->predict(sample, result);
			cout << sample << " -> ";// << result << endl;
			cout << result.at<float>(0, 0);
			cout << endl;
		}
		return mlp;
	}
	cv::Mat src[5];
	void CollectImagesFromFolder()
	{
		for (int index = 0; index < 5; index++)
		{
			cv::String filePath = cv::format("media/HamNoSysLoc15/%d.png", index);
			src[index] = cv::imread(filePath);
		}
	}
	//https://stackoverflow.com/questions/34547400/opencv-3-0-cant-load-neural-network
	float PredictToFloat(Ptr<ANN_MLP> mlp, std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumsPtr)
	{
		float Features[54];
		vector<float> vv;
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
			int i = 0;
			for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
			{
				Features[i] = poseKeypoints[{0, bodyPart, 0}];
				Features[i + 1] = poseKeypoints[{0, bodyPart, 1}];
				Features[i + 2] = poseKeypoints[{0, bodyPart, 2}];
				i = i + 3;
			}
			vector<float> v(Features, Features + sizeof Features / sizeof Features[0]);
			
			vv = coordinatToAngle(FeatureSelection(v));
		}
		else
			op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
		float inputTrainingDataArray[4]; int nn = 4;
		for (int i = 0; i < nn; i++)
		{
			inputTrainingDataArray[i] = vv[i];
		}
		Mat sample = Mat(1, 4, CV_32F, inputTrainingDataArray);
		Mat result;
		mlp->predict(sample, result);
		return result.at<float>(0, 0);
	}

	int Predict(int argc, char *argv[], std::string modelname)
	{
		gflags::ParseCommandLineFlags(&argc, &argv, true);
		OpenPoseWrapper opw;
		op::log("Collect images from folder...");
		CollectImagesFromFolder();
		try
		{
			op::log("Starting machine learning demo...", op::Priority::High);
			op::log("Reading from NN model file: " + modelname, op::Priority::High);
			//Ptr<ANN_MLP> mlp = ML2(ReadFromCsvToVector(filename), filename);
			Ptr<ANN_MLP> mlp = Algorithm::load<ANN_MLP>(modelname);
			op::log("Starting OpenPose demo...", op::Priority::High);
			const auto opTimer = op::getTimerInit();
			// Configuring OpenPose
			op::log("Configuring OpenPose...", op::Priority::High);
			op::Wrapper opWrapper{ op::ThreadManagerMode::AsynchronousOut };
			opw.configureWrapper(opWrapper);
			// Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
			op::log("Starting thread(s)...", op::Priority::High);
			opWrapper.start();
			// User processing
			bool userWantsToExit = false;
			//int i=1;
			while (!userWantsToExit)
			{
				// Pop frame
				std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
				if (opWrapper.waitAndPop(datumProcessed))
				{
					if (!FLAGS_no_display)
						userWantsToExit = opw.display(datumProcessed);
					//ѕредсказывание
					float pred;
					pred = PredictToFloat(mlp, datumProcessed);
					//if (datumProcessed != nullptr && !datumProcessed->empty())
					//op::log((int)round(pred));
					cv::imshow("test", src[(int)round(pred)]);//ѕервый индекс массива должен максимальное разрешение всех
				}
				// If OpenPose finished reading images
				else if (!opWrapper.isRunning())
					break;
				// Something else happened
				else
					op::log("Processed datum could not be emplaced.", op::Priority::High);
			}
			op::log("Stopping thread(s)", op::Priority::High);
			opWrapper.stop();
			// Measuring total time
			op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);
			// Return
			return 0;
		}
		catch (const std::exception& e)
		{
			return -1;
		}
	}
	ML()
	{
	}
	~ML()
	{
	}
};

