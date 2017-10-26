#include "SystemCode.h"

SystemCode::SystemCode(const unsigned int &code, const string &info)
:code(code),info(info){
}

SystemCode::~SystemCode() {
}

const unsigned int& SystemCode::getCode() const{
	return this->code;
}

const string& SystemCode::getInfo() const{
	return this->info;
}
