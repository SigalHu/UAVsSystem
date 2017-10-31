//#include <iostream>
//#include "common.h"
//#include "cuda.h"
//#include "DeviceVector.cpp"
//#include "dev_noise.cuh"
//using namespace std;
//
//int main(){
//	DeviceVector<float> vc1(0,10,1);
//	DeviceVector<float> vc2(0, 10,1);
//	for (float aa : vc1){
//		cout << aa << " ";
//	}
//	cout << endl;
//	for (float aa : vc2){
//		cout << aa << " ";
//	}
//	cout << endl;
//
//	cudaNoiseGeneWithSoS << <1, vc1.size() >> >(raw_pointer_cast(vc1.data()), raw_pointer_cast(vc2.data()), vc1.size(), vc1.size(), 0/*time(NULL)*/, 1, 1, 1, 1);
//
//	for (float aa : vc1){
//		cout << aa << " ";
//	}
//	cout << endl;
//	for (float aa : vc2){
//		cout << aa << " ";
//	}
//	cout << endl;
//	cin.get();
//}