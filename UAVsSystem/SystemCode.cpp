#include "SystemCode.h"

SystemCode::SystemCode(const unsigned int &code, const std::string &info)
:code(code),info(info){
}

SystemCode::~SystemCode() {
}

const unsigned int& SystemCode::getCode() const{
	return this->code;
}

const std::string& SystemCode::getInfo() const{
	return this->info;
}
