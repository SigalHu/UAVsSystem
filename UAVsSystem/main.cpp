#include <iostream>
#include "common.h"
#include "service.h"
using namespace std;

template<class _T>
class A{
public:
	std::string getClassName() const{
		return MacroUtils_ClassName(A);
	}
};

int main(){
	A<int> service;
	cout << service.getClassName() << endl;
	cin.get();
}



//#include <iostream>
//#include <time.h>
//#include "cuda.h"
//#include "common.h"
//using namespace std;
//
//int main(){
//	DeviceVector<float> vc1(0,10, 0);
//	DeviceVector<float> vc2(0,10, 0);
//	for (float aa : vc1){
//		cout << aa << " ";
//	}
//	cout << endl;
//	for (float aa : vc2){
//		cout << aa << " ";
//	}
//	cout << endl;
//
//	cudaNoiseGeneWithSoS<<<1,vc1.size()>>>(raw_pointer_cast(vc1.data()), raw_pointer_cast(vc2.data()), vc1.size(), vc1.size(), time(NULL), 1, 1, 1, 1);
//
//	for (float aa : vc1){
//		cout << aa << " ";
//	}
//	cout << endl;
//	for (float aa : vc2){
//		cout << aa << " ";
//	}
//	cout << endl;
//
////	CudaNoiseService service(1000,1000);
////	cout << service.toString() << endl;
//
///*	clock_t start, stop;
//	bool isSucceed;
//	const float fs = pow(2,10);
//	const float time_spend = pow(2,10);
//	const unsigned int len = (unsigned int)(fs*time_spend);
//	valarray<valarray<float>> noise(valarray<float>(len), 2);
//
//	start = clock();
//	isSucceed = CudaSoSUtils::cudaNoiseGeneWithSoS(&noise[0][0], &noise[1][0], fs, time_spend, 2);
//	stop = clock();
//
//	if (isSucceed){
//		FILE *fp = fopen("F:/SigalHu/UAVsSystem/Matlab/noise.bin", "wb");
//		if (fp){
//			fwrite(&noise[0][0], sizeof(float), noise[0].size(), fp);
//			fwrite(&noise[1][0], sizeof(float), noise[1].size(), fp);
//			fclose(fp);
//		}
//
//		//for (valarray<float> &ii : noise){
//		//	for (float &jj : ii){
//		//		cout << jj << " ";
//		//	}
//		//	cout << endl << "**********************" << endl;
//		//}
//		cout << "调用成功！" << endl;
//	}
//	else{
//		cout << "调用失败！" << endl;
//	}
//
//	cout << "所花时间：" << stop - start << "ms" << endl;
//	cin.get();
//
//	start = clock();
//	isSucceed = CudaRandUtils::cudaNoiseGene(&noise[0][0], &noise[1][0], noise[0].size(), 0, 1);
//	stop = clock();
//
//	if (isSucceed){
//		//FILE *fp = fopen("F:/SigalHu/UAVsSystem/Matlab/noise.bin", "wb");
//		//if (fp){
//		//	fwrite(&noise[0][0], sizeof(float), noise[0].size(), fp);
//		//	fwrite(&noise[1][0], sizeof(float), noise[1].size(), fp);
//		//	fclose(fp);
//		//}
//
//		//for (valarray<float> &ii : noise){
//		//	for (float &jj : ii){
//		//		cout << jj << " ";
//		//	}
//		//	cout << endl << "**********************" << endl;
//		//}
//		cout << "调用成功！" << endl;
//	}
//	else{
//		cout << "调用失败！" << endl;
//	}
//
//	cout << "所花时间：" << stop - start << "ms" <<endl;*/
//	cin.get();
//
//	return 0;
//}