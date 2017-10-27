#include "SystemException.h"

string SystemException::formatException(const SystemCode &code, const string &className, const string &functionName, const string &variableName, const string &detail) const{
	return StringUtils::format(code.getInfo(), className.c_str(), functionName.c_str(), variableName.c_str(), detail.c_str());
}

SystemException::SystemException(const SystemCode &code, const string &info)
:_Code(code), exception(info.c_str()){
}

SystemException::SystemException(const SystemCode &code, const string &className, 
	const string &functionName, const string &variableName, const string &detail)
	: _Code(code), exception(formatException(code, className, functionName, variableName, detail).c_str()){
}


SystemException::~SystemException(){
}

unsigned int SystemException::code() const{
	return this->_Code.getCode();
}
