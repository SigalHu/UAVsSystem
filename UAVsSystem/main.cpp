#include <iostream>
#include <time.h>
#include <valarray>
#include "noise.h"

using namespace std;

int main()
{
	clock_t start, stop;
	bool isSucceed;
	const float fs = pow(2,10);
	const float time_spend = pow(2,16);
	const unsigned int len = (unsigned int)(fs*time_spend);
	valarray<valarray<float>> noise(valarray<float>(len), 2);

	start = clock();
	isSucceed = cudaNoiseGeneWithSoS(&noise[0][0], &noise[1][0], fs, time_spend, 1, 3);
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
	isSucceed = cudaNoiseGene(&noise[0][0], &noise[1][0], noise[0].size(), 0, 1);
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

	cout << "所花时间：" << stop - start << "ms" <<endl;
	cin.get();

	return 0;
}