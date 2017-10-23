#include "SystemException.h"

SystemException::SystemException(const SystemCode &code, const string &info)
:_Code(code), exception(info.c_str()){
}


SystemException::~SystemException(){
}

unsigned int SystemException::code() const{
	return this->_Code.getCode();
}
