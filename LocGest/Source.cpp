#include "FileWork.h"
#include "ml.h"
#include "iostream"
void main(int argc, char *argv[])
{
	//----------------------------------------------------------------------------------------
	//Тестирование примера
	//OpenPoseWrapper OpenPose;
	//OpenPose.OpenPoseApiEx();
	//----------------------------------------------------------------------------------------
	FileWork csv;
	char *as = argv[1];
	//----------------------------------------------------------------------------------------
	//Ввод различных объектов места исполнения жеста и различных меток классов в csv-файл с помощью веб-камеры;
	//cmd: LocGest.exe -csv NameCsvFile label
	//cmd example: LocGest.exe -csv name.csv 0
	char *as1;
	char par1[] = "-csv";
	as1 = par1;
	if (!strcmp(as, as1) && argc == 4)
	{
		op::log("Writing to csv...");
		csv.OpenPoseApiWriteToCsv(argv[2], argv[3], 30);
	}
	//----------------------------------------------------------------------------------------
	//Анимационное представление точек места исполнения жеста из csv-файла объектов различных места исполнения жеста;
	//cmd: LocGest.exe -outcsv NameCsvFile 
	//cmd example: LocGest.exe -outcsv name.csv
	char *as2;
	char par2[] = "-outcsv";
	as2 = par2;
	if (!strcmp(as, as2) && argc == 3) csv.DisplayFromCsvFile(argv[2]);
	//----------------------------------------------------------------------------------------
	//Машинное обучение из csv-файла и сохранение модели распознавания места исполнения жеста(формат модели .yml);
	//cmd: LocGest.exe -ml NameCsvFile
	//cmd example: LocGest.exe -ml name.csv
	char *as3;
	char par3[] = "-ml";
	as3 = par3;
	ML ml;
	if (!strcmp(as, as3) && argc == 3) ml.Learning(csv.ReadFromCsvToVector(argv[2]), argv[2]);
	//----------------------------------------------------------------------------------------
	//Распознавание места исполнения жеста с моделью его;
	//cmd: LocGest.exe --video NameMp4File NamePredictModelfile
	//cmd example: LocGest.exe --video g3334.mp4 name.yml
	char *as4;
	char par4[] = "--video";
	as4 = par4;
	if (!strcmp(as, as4) && argc == 4)ml.Predict(argc, argv, argv[3]);
}