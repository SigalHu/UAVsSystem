#include "HostVector.h"

template<class _T, class _Alloc>
HostVector<_T, _Alloc>::HostVector(){
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>::HostVector(const size_t &n)
:host_vector(n){
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>::HostVector(const size_t &n, const _T &value)
: host_vector(n, value){
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>::HostVector(const HostVector &v)
: host_vector(v){
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>::HostVector(HostVector &&v)
: host_vector(move(v)){
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>& HostVector<_T, _Alloc>::operator=(const HostVector &v){
	host_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>& HostVector<_T, _Alloc>::operator=(HostVector &&v){
	host_vector::operator=(move(v));
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>::HostVector(const HostVector<_OtherT, _OtherAlloc> &v) 
: host_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>& HostVector<_T, _Alloc>::operator=(const HostVector<_OtherT, _OtherAlloc> &v){
	host_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>::HostVector(const std::vector<_OtherT, _OtherAlloc> &v) 
: host_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>& HostVector<_T, _Alloc>::operator=(const std::vector<_OtherT, _OtherAlloc> &v){
	host_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>::HostVector(const DeviceVector<_OtherT, _OtherAlloc> &v)
: host_vector(v){
}

template<class _T, class _Alloc>
template<class _OtherT, class _OtherAlloc>
HostVector<_T, _Alloc>& HostVector<_T, _Alloc>::operator=(const DeviceVector<_OtherT, _OtherAlloc> &v){
	host_vector::operator=(v);
	return *this;
}

template<class _T, class _Alloc>
template<class InputIterator>
HostVector<_T, _Alloc>::HostVector(InputIterator first, InputIterator last) 
: host_vector(first, last){
}