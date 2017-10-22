#include <iostream>
#include <time.h>
#include <valarray>
#include <string>
#include "UtilsDeclaration.h"
#include "ServiceDeclaration.h"
using namespace std;


template<class _T1>
void call(_T1 cudaTask){
	static_assert(is_base_of<CudaTask, _T1>::value, "cudaTask must be a subclass of CudaTask.");

	cudaTask();
}

class CudaAA :public CudaTask{
public:
	CudaAA(dim3 a,dim3 b):CudaTask(a,b){}

	void operator()(){
		cout << "123" << endl;
	}
};

int main()
{
	call(CudaAA(dim3(1, 1, 1), dim3(1, 1, 1)));

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