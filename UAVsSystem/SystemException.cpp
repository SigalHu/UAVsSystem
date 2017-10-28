#include "StringUtils.h"
#include "SystemException.h"

SystemException::SystemException(const SystemCode &code, const std::string &className,
	const std::string &functionName, const std::string &variableName, const std::string &detail)
	: _Code(code), _ClassName(className), _FunctionName(functionName), _Detail(detail){
}


SystemException::~SystemException(){
}

unsigned int SystemException::code() const{
	return this->_Code.getCode();
}

const char* SystemException::what() const{
	return StringUtils::format(this->_Code.getInfo(), this->_ClassName.c_str(), this->_FunctionName.c_str(), this->_VariableName.c_str(), this->_Detail.c_str()).c_str();
}

std::string SystemException::getClassName() const{
	return this->_ClassName;
}

void SystemException::setClassName(const std::string &className){
	this->_ClassName = className;
}

std::string SystemException::getFunctionName() const{
	return this->_FunctionName;
}

void SystemException::setFunctionName(const std::string &functionName){
	this->_FunctionName = functionName;
}

std::string SystemException::getVariableName() const{
	return this->_VariableName;
}

void SystemException::setVariableName(const std::string &variableName){
	this->_VariableName = variableName;
}

std::string SystemException::getDetail() const{
	return this->_Detail;
}

void SystemException::setDetail(const std::string &detail){
	this->_Detail = detail;
}