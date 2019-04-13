#pragma once
#include "OpenPoseWrapper.h"
class FileWork
{
public:
	FileWork()
	{
	}
	void WriteToCsv(std::string filename, std::string s)
	{
		std::ofstream myfile;
		myfile.open(filename, std::ios_base::app);
		myfile << s << "\n";
		myfile.close();
	}
	std::string PoseDatumvectorToCsvstr(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, std::string label)
	{
		std::string strcsv = "";
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
			//op::log(poseKeypoints.getSize(1));
			for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
			{
				if (strcsv == "")
				{
					strcsv = std::to_string(poseKeypoints[{0, bodyPart, 0}]) + "," + std::to_string(poseKeypoints[{0, bodyPart, 1}]) + "," + std::to_string(poseKeypoints[{0, bodyPart, 2}]);
					//op::log("1: "+ std::to_string(bodyPart));
				}
				else
				{
					strcsv = strcsv + "," + std::to_string(poseKeypoints[{0, bodyPart, 0}]) + "," + std::to_string(poseKeypoints[{0, bodyPart, 1}]) + "," + std::to_string(poseKeypoints[{0, bodyPart, 2}]);
					//op::log("0: "+ std::to_string(bodyPart));
					if (bodyPart == poseKeypoints.getSize(1) - 1)strcsv = strcsv + "," + label;
				}
			}
			//strcsv = strcsv + "," + label;
			//op::log(strcsv);
		}
		else
			op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
		return strcsv;
	}
	int i=0;
	int OpenPoseApiWriteToCsv(std::string filename, std::string label, int count)
	{		
		OpenPoseWrapper OpenPose;
		try
		{
			op::log("Starting OpenPose demo...", op::Priority::High);
			const auto opTimer = op::getTimerInit();
			// Configuring OpenPose
			op::log("Configuring OpenPose...", op::Priority::High);
			op::Wrapper opWrapper{ op::ThreadManagerMode::AsynchronousOut };
			OpenPose.configureWrapper(opWrapper);
			// Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
			op::log("Starting thread(s)...", op::Priority::High);
			opWrapper.start();
			// User processing
			// UserOutputClass userOutputClass;
			bool userWantsToExit = false;
			//int i=1;
			while (!userWantsToExit)
			{
				// Pop frame
				std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
				if (opWrapper.waitAndPop(datumProcessed))
				{
					if (!FLAGS_no_display)
						userWantsToExit = OpenPose.display(datumProcessed);
					if (i < count)
					{
						op::log("CountIndex = " + std::to_string(i));
						WriteToCsv(filename, PoseDatumvectorToCsvstr(datumProcessed, label));
						i++;
					}
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
	std::vector<std::vector<double>> ReadFromCsvToVector(std::string filename)
	{
		std::ifstream in(filename);
		std::vector<std::vector<double>> fields;
		if (in) {
			std::string line;
			while (getline(in, line)) {
				std::stringstream sep(line);
				std::string field;
				fields.push_back(std::vector<double>());
				while (getline(sep, field, ',')) {
					fields.back().push_back(stod(field));
				}
			}
		}
		return fields;
	}
	void TextShow(int label, cv::Mat img)
	{
		std::string text = "Label";
		int fontFace = cv::FONT_HERSHEY_COMPLEX;
		double fontScale = 1;
		int thickness = 3;
		int baseline = 0;
		cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
		cv::Point textOrg(5, 30);
		cv::putText(img, cv::format("label = %i", label), textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
	}	
	void VectorToShow(std::vector<std::vector<double>> datumsPtr, int i)
	{
		int ss = 0;
		cv::Mat img(480, 1024, CV_8UC3, cv::Scalar::all(0));
		for (int j = 0; j < datumsPtr[0].size(); j++)
		{
			int x, y, p;
			if (datumsPtr[0].size() - 1 == j)TextShow(datumsPtr[i][j], img);
			else
			{
				if (datumsPtr[i][j])
				{
					if (j % 3 == 0)
						x = datumsPtr[i][j];
					if (j % 3 == 1)
						y = datumsPtr[i][j];
					if (j % 3 == 2)
						if (datumsPtr[i][j] > 0.4)
						{
							cv::circle(img, cv::Point(x, y), 4, cv::Scalar(0, 25, 255), 2);
							ss++;
						}
				}
			}
			cv::imshow("test", img);
		}
	}
	//#include <opencv2/core.hpp>
	void DisplayFromCsvFile(std::string filename)
	{
		std::vector<std::vector<double>> a = ReadFromCsvToVector(filename);
		int size = ReadFromCsvToVector(filename).size();
		op::log(size);
		for (int i = 0; ; i++)
		{
			VectorToShow(a, i);
			if (i > size - 2)i = 0;
			if (cv::waitKey(10) > 0) break;
		}
	}

	~FileWork()
	{

	}
};

