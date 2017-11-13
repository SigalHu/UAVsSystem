#include <iostream>
#include <ctime>
//#include "DeviceVector.hpp"
//#include "HostVector.hpp"
//#include "CudaSoSUtils.h"
using namespace std;

int main(){
	//try{
	//	clock_t start, stop;
	//	bool isSucceed;
	//	const float fs = 1e6;
	//	const float time_spend = 1e3;
	//	const unsigned int len = (unsigned int)(fs*time_spend);
	//	DeviceVector<float> devNoiseI(len);
	//	DeviceVector<float> devNoiseQ(len);

	//	start = clock();
	//	CudaSoSUtils::noiseGene(devNoiseI, devNoiseQ, fs);
	//	stop = clock();

	//	HostVector<float> hostNoiseI = devNoiseI;
	//	HostVector<float> hostNoiseQ = devNoiseQ;
	//	FILE *fp = fopen("F:/SigalHu/UAVsSystem/Matlab/noise.bin", "wb");
	//	if (fp){
	//		fwrite(hostNoiseI.data(), sizeof(float), hostNoiseI.size(), fp);
	//		fwrite(hostNoiseQ.data(), sizeof(float), hostNoiseQ.size(), fp);
	//		fclose(fp);
	//	}
	//	cout << "调用成功！" << endl;
	//	cout << "所花时间：" << stop - start << "ms" << endl;
	//}
	//catch (SystemException &ex){
	//	cout << "调用失败！" << endl;
	//	cout << ex.what() << endl;
	//}

	cin.get();
	return 0;
}