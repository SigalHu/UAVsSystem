#include <iostream>
#include "cuda.h"
using namespace std;
using namespace thrust;

class A{
public:
	A(){
		cout << "A()" << endl;
	}

	A(const A &a){
		cout << "A(const A &a)" << endl;
	}

	A(A &&a){
		cout << "A(A &&a)" << endl;
	}

	~A(){
		cout << "~A()" << endl;
	}
};

A fun(){
	return A();
}

int main(){
	A a;
	try{
		cout << MacroUtils_ClassName(a) << MacroUtils_FunctionName() << MacroUtils_VariableName(a) << endl;
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			StringUtils::format(SystemCodeEnum::OUT_OF_RANGE.getInfo(),
			MacroUtils_ClassName(a).c_str(), MacroUtils_FunctionName().c_str(), MacroUtils_VariableName(a).c_str(),
			MacroUtils_VariableName(a).append(" must be less than the number of GPU devices.").c_str()));
	}
	catch (SystemException &ex){
		cout << ex.what() << endl;
	}

//	CudaNoiseService service(1000,1000);
//	cout << service.toString() << endl;

/*	clock_t start, stop;
	bool isSucceed;
	const float fs = pow(2,10);
	const float time_spend = pow(2,10);
	const unsigned int len = (unsigned int)(fs*time_spend);
	valarray<valarray<float>> noise(valarray<float>(len), 2);

	start = clock();
	isSucceed = CudaAlgorithmUtils::cudaNoiseGeneWithSoS(&noise[0][0], &noise[1][0], fs, time_spend, 2);
	stop = clock();

	if (isSucceed){
		FILE *fp = fopen("F:/SigalHu/UAVsSystem/Matlab/noise.bin", "wb");
		if (fp){
			fwrite(&noise[0][0], sizeof(float), noise[0].size(), fp);
			fwrite(&noise[1][0], sizeof(float), noise[1].size(), fp);
			fclose(fp);
		}

		//for (valarray<float> &ii : noise){
		//	for (float &jj : ii){
		//		cout << jj << " ";
		//	}
		//	cout << endl << "**********************" << endl;
		//}
		cout << "调用成功！" << endl;
	}
	else{
		cout << "调用失败！" << endl;
	}

	cout << "所花时间：" << stop - start << "ms" << endl;
	cin.get();

	start = clock();
	isSucceed = CudaCommonUtils::cudaNoiseGene(&noise[0][0], &noise[1][0], noise[0].size(), 0, 1);
	stop = clock();

	if (isSucceed){
		//FILE *fp = fopen("F:/SigalHu/UAVsSystem/Matlab/noise.bin", "wb");
		//if (fp){
		//	fwrite(&noise[0][0], sizeof(float), noise[0].size(), fp);
		//	fwrite(&noise[1][0], sizeof(float), noise[1].size(), fp);
		//	fclose(fp);
		//}

		//for (valarray<float> &ii : noise){
		//	for (float &jj : ii){
		//		cout << jj << " ";
		//	}
		//	cout << endl << "**********************" << endl;
		//}
		cout << "调用成功！" << endl;
	}
	else{
		cout << "调用失败！" << endl;
	}

	cout << "所花时间：" << stop - start << "ms" <<endl;*/
	cin.get();

	return 0;
}